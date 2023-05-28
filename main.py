#!/usr/bin/env python3

from device import *

from PicoscenesToolbox.picoscenes import Picoscenes

from tqdm import tqdm

import numpy as np

from pathlib import Path

import argparse


def pico2Numpy(picoRaw: list[dict], type: str) -> np.ndarray:
  assert type in ("csi", "mag", "phase")

  map = {"csi": "CSI", "mag": "Mag", "phase": "Phase"}
  dataList: list[np.ndarray] = []

  for raw in tqdm(picoRaw):
    # Parse CSI
    deviceId: int = raw["CSI"]["DeviceType"]
    cbw: int = raw["CSI"]["CBW"]
    device: str = deviceMap[deviceId]
    subcarrierIndex = subcarrierList[device][str(cbw)]

    nS: int = raw["CSI"]["numTones"]
    nTx: int = raw["CSI"]["numTx"]
    nRx: int = raw["CSI"]["numRx"]
    shape: tuple = (nS, nTx, nRx)

    # PicoscenesToolbox always interpolate the missing subcarrier
    # Extract non-interpolated CSI
    picoSubcarrierIndex: list[int] = raw["CSI"]["SubcarrierIndex"]
    picoData = np.array(raw["CSI"][map[type]]).reshape(shape)
    realData: list[np.ndarray] = []

    for idx in subcarrierIndex:
      realSubIndex = picoSubcarrierIndex.index(idx)
      realData.append(picoData[realSubIndex])

    realNdarray = np.array(realData)
    dataList.append(realNdarray)

  return np.array(dataList)


def parseCli():
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", type=str, metavar="", default="in")
  parser.add_argument("-o", "--output", type=str, metavar="", default="out")
  args = parser.parse_args()
  return args


def parsePico(inPath: Path, outDir: Path):
  outDir.mkdir(parents=True, exist_ok=True)

  picoFrames = Picoscenes(str(inPath))

  print(f"Saving timestamp...")
  timestampList: list[np.datetime64] = []
  for raw in tqdm(picoFrames.raw):
    # Parse timesteamp
    rawTimesteampNs: int = raw["RxSBasic"]["systemns"]
    timestamp = np.datetime64(rawTimesteampNs, "ns")
    timestampList.append(timestamp)
  np.save(str(outDir / "timestamp.npy"), timestampList)

  print(f"Saving CSI...")
  saveCsi(outDir, picoFrames.raw, "csi")
  print(f"Saving mag...")
  saveCsi(outDir, picoFrames.raw, "mag")
  print(f"Saving phase...")
  saveCsi(outDir, picoFrames.raw, "phase")


def saveCsi(outDir: Path, picoRaw: list[dict], type: str):
  outPath = outDir / f"{type}.npy"
  dataArray = pico2Numpy(picoRaw, type)
  np.save(str(outPath), dataArray)


if __name__ == "__main__":
  scriptPath = Path(__file__).parent

  cliArgs = parseCli()
  inDir = Path(cliArgs.input)
  outDir = Path(cliArgs.output)

  for rawPath in inDir.glob("*.csi"):
    print(f"Parsing {rawPath.name}...")
    parsePico(rawPath, outDir / rawPath.stem)
