#!/usr/bin/env python3
from tools.Header import *

def main() -> int:
    parser = argparse.ArgumentParser(description='Robot Simulation')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Name of config file in config directory')
    parser.add_argument('--param', nargs=2, action='append',
                       metavar=('parameter', 'value'),
                       help='Override config parameters: --param parameter value')
    args = parser.parse_args()
    return run_simulation(args)

if __name__ == "__main__":
    sys.exit(main())