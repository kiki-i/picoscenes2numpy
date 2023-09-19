#!/usr/bin/env python3

from device import *
from parsecli import *

from PicoscenesToolbox.picoscenes import Picoscenes
from tqdm import tqdm
import numpy as np

from pathlib import Path


def picoFrame2numpy(frameRaw: dict, type: str, interpolate: bool):
  assert type in ("csi", "amp", "phase", "timestamp")

  # Parse timestamp
  if type == "timestamp":
    rawTimesteampNs: int = frameRaw["RxSBasic"]["systemns"]
    timestamp = np.datetime64(rawTimesteampNs, "ns")
    return timestamp

  # Parse CSI
  else:
    typeMap = {"csi": "CSI", "amp": "Mag", "phase": "Phase"}
    deviceId: int = frameRaw["CSI"]["DeviceType"]
    cbw: int = frameRaw["CSI"]["CBW"]
    device: str = deviceMap[deviceId]

    nS: int = frameRaw["CSI"]["numTones"]
    nTx: int = frameRaw["CSI"]["numTx"]
    nRx: int = frameRaw["CSI"]["numRx"]
    shape: tuple = (nS, nTx, nRx)

    ## PicoscenesToolbox always interpolate the missing subcarrier
    ## Extract non-interpolated CSI
    picoCsi = np.array(frameRaw["CSI"][typeMap[type]]).reshape(shape)

    if interpolate:
      return picoCsi
    else:
      picoSubcarrierIndex: list[int] = frameRaw["CSI"]["SubcarrierIndex"]
      realSubcarrierIndex = subcarrierList[device][str(cbw)]
      realCsi: list[np.ndarray] = []
      for idx in realSubcarrierIndex:
        realSubcarrier = picoSubcarrierIndex.index(idx)
        realCsi.append(picoCsi[realSubcarrier])
      return np.array(realCsi)


def pico2Numpy(picoRaw: list[dict],
               types: set[str],
               interpolate: bool = False) -> dict[str, np.ndarray]:

  outputDict: dict[str, np.ndarray] = dict.fromkeys(types)

  # Parse each frame along timestep
  for raw in tqdm(picoRaw, leave=False, desc="Frames"):
    for type in types:
      frameList = []
      frame = picoFrame2numpy(raw, type, interpolate)
      frameList.append(frame)
      outputDict[type] = np.array(frameList)

  return outputDict


def parsePico(inPath: Path, outDir: Path, types: set[str]):
  outDir.mkdir(parents=True, exist_ok=True)

  outputDict = pico2Numpy(Picoscenes(str(inPath)).raw, types)
  for type, dataArray in outputDict.items():
    filename = inPath.with_suffix(f".{type}.npy").name
    np.save(outDir / filename, dataArray)


if __name__ == "__main__":
  scriptPath = Path(__file__).parent

  config = parseCli()
  inDir = Path(config.inDir)
  outDir = Path(config.outDir)

  for rawPath in tqdm(set(inDir.glob("*.csi")), desc="Files"):
    parsePico(rawPath, outDir, config.types)
