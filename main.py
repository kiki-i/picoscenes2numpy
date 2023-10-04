#!/usr/bin/env python3

from device import *
from parsecli import *

from PicoscenesToolbox.picoscenes import Picoscenes
from tqdm import tqdm
import numpy as np

from pathlib import Path


def picoFrame2numpy(frameRaw: dict, dataType: str,
                    interpolate: bool) -> np.datetime64 | np.ndarray:
  assert dataType in ("csi", "amp", "phase", "timestamp")

  # Parse timestamp
  if dataType == "timestamp":
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
    picoCsi = np.array(frameRaw["CSI"][typeMap[dataType]]).reshape(shape)

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
               types: tuple,
               interpolate: bool = False) -> dict[str, np.ndarray]:

  outputByType: dict[str, list[np.datetime64 | np.ndarray]] = {}
  outputByTypeNp: dict[str, np.ndarray] = dict.fromkeys(types)

  for dataType in types:
    outputByType[dataType] = []

  # Parse each frame along reversed time axis
  n = len(picoRaw)
  with tqdm(total=len(picoRaw), leave=False, desc="Frames") as pbar:
    for _ in range(n):
      raw = picoRaw.pop()
      for dataType in types:
        frame = picoFrame2numpy(raw, dataType, interpolate)
        outputByType[dataType].append(frame)
      pbar.update()

  for dataType in tuple(outputByType.keys()):
    data = outputByType.pop(dataType)
    outputByTypeNp[dataType] = np.array(reversed(data))
  return outputByTypeNp


def saveNumpy(inPath: Path, outDir: Path, types: tuple):
  outDir.mkdir(parents=True, exist_ok=True)

  outputByType = pico2Numpy(Picoscenes(str(inPath)).raw, types)
  for dataType in tuple(outputByType.keys()):
    data = outputByType.pop(dataType)
    filename = inPath.with_suffix(f".{dataType}.npy").name
    np.save(outDir / filename, data)


if __name__ == "__main__":
  scriptPath = Path(__file__).parent

  config = parseCli()
  inDir = Path(config.inDir)
  outDir = Path(config.outDir)

  for rawPath in tqdm(tuple(inDir.glob("*.csi")), desc="Files"):
    saveNumpy(rawPath, outDir, config.types)
