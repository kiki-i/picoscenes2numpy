#!/usr/bin/env python3

from PicoscenesToolbox.picoscenes import Picoscenes

import numpy as np

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import argparse
import pickle
import json

deviceMap = {
    0x2000: "AX200",
    0x2100: "AX210",
    0x5300: "IWL5300",
    0x9300: "QCA9300",
    0x1234: "USRP"
}


def parseCli():
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", type=str, metavar="", default="in")
  parser.add_argument("-o", "--output", type=str, metavar="", default="out")
  args = parser.parse_args()
  return args


def parseCsi(inPath: Path, outDir: Path):
  rawList: list = []

  csiList: list[np.ndarray] = []
  magList: list[np.ndarray] = []
  phaseList: list[np.ndarray] = []
  timestampList: list[datetime] = []

  frames = Picoscenes(str(inPath))
  startTime: datetime = None
  startTimeStr: str = ""

  with open("subcarrier-index.json", "rt") as f:
    deviceIndex = json.load(f)

  for raw in tqdm(frames.raw):
    rawDevice: int = raw["CSI"]["DeviceType"]
    cbw: int = raw["CSI"]["CBW"]
    device: str = deviceMap[rawDevice]
    subcarrierIndex = tuple(deviceIndex[device][str(cbw)])

    nSub: int = raw["CSI"]["numTones"]
    nTx: int = raw["CSI"]["numTx"]
    nRx: int = raw["CSI"]["numRx"]
    sciShape: tuple = (nSub, nTx, nRx)

    # PicoscenesToolbox always interpolate the missing subcarrier
    interpolatedSubIndex: list[int] = raw["CSI"]["SubcarrierIndex"]
    interpolatedCsi = np.array(raw["CSI"]["CSI"]).reshape(sciShape)
    interpolatedMag = np.array(raw["CSI"]["Mag"]).reshape(sciShape)
    interpolatedPhase = np.array(raw["CSI"]["Phase"]).reshape(sciShape)

    # Extract non-interpolated CSI
    realCsi: list[np.ndarray] = []
    realMag: list[np.ndarray] = []
    realPhase: list[np.ndarray] = []
    for idx in subcarrierIndex:
      realSubIndex = interpolatedSubIndex.index(idx)
      realCsi.append(interpolatedCsi[realSubIndex])
      realMag.append(interpolatedMag[realSubIndex])
      realPhase.append(interpolatedPhase[realSubIndex])

    rawList.append(raw)

    csiList.append(np.concatenate(realCsi))
    magList.append(np.concatenate(realMag))
    phaseList.append(np.concatenate(realPhase))

    rawTimesteampNs: int = raw["RxSBasic"]["systemns"]
    rawTimesteampS = rawTimesteampNs / 1e9
    timestamp = datetime.fromtimestamp(rawTimesteampS).astimezone()
    timestampList.append(timestamp)

    if startTime == None:
      startTime = timestamp
      startTimeStr = startTime.isoformat().replace(":", ";")

  print("Saving...")

  outSubDir = outDir / f"[{startTimeStr}]"
  outSubDir.mkdir(parents=True, exist_ok=True)
  outCsiPath = outSubDir / "csi.npy"
  outMagPath = outSubDir / "mag.npy"
  outPhasePath = outSubDir / "phase.npy"
  outTimestamp = outSubDir / "timestamp.txt"
  outRawPath = outSubDir / "raw.pkl"

  with open(outRawPath, "wb") as file:
    pickle.dump(rawList, file)

  np.save(outCsiPath, csiList)
  np.save(outMagPath, magList)
  np.save(outPhasePath, phaseList)

  with open(outTimestamp, "wt") as timestampFile:
    for timestamp in timestampList:
      timestampFile.write(timestamp.isoformat() + "\n")


if __name__ == "__main__":
  scriptPath = Path(__file__).parent

  cliArgs = parseCli()
  inDir = Path(cliArgs.input)
  outDir = Path(cliArgs.output)

  for rawPath in inDir.glob("*.csi"):
    print(f"Processing {rawPath.name}...")
    parseCsi(rawPath, Path(outDir))
