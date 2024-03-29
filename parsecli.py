from dataclasses import dataclass
import argparse


@dataclass
class Config:
  inDir: str
  outDir: str
  types: frozenset


def parseCli() -> Config:
  description = "Convert PicoScenes CSI data to numpy ndarray"
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument(
    "-i",
    "--input",
    type=str,
    metavar="",
    default="in",
    help='Specify input directory, default = "in"',
  )
  parser.add_argument(
    "-o",
    "--output",
    type=str,
    metavar="",
    default="out",
    help='Specify output directory, default = "out"',
  )
  parser.add_argument(
    "-c",
    "--csi",
    metavar="",
    action="store_const",
    const=True,
    default=False,
    help="Enable complex CSI output",
  )
  parser.add_argument(
    "-m",
    "--magnitude",
    metavar="",
    action="store_const",
    const=True,
    default=False,
    help="Enable magnitude output",
  )
  parser.add_argument(
    "-p",
    "--phase",
    metavar="",
    action="store_const",
    const=True,
    default=False,
    help="Enable phase output (with cyclic shift delay removed)",
  )
  parser.add_argument(
    "-t",
    "--timestamp",
    metavar="",
    action="store_const",
    const=True,
    default=False,
    help="Enable timestamp output",
  )
  args = parser.parse_args()

  types = set()
  if args.csi:
    types.add("csi")
  if args.magnitude:
    types.add("mag")
  if args.phase:
    types.add("phase")
  if args.timestamp:
    types.add("timestamp")

  return Config(inDir=args.input, outDir=args.output, types=frozenset(types))
