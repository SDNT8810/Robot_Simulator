#!/usr/bin/env python3
from tools.Header import *

def main() -> int:
    parser = argparse.ArgumentParser(description='Robot Simulation')
    parser.add_argument('--config', type=str, default='',
                       help='Name of config file in config directory (leave blank to use default+temporary merge)')
    parser.add_argument('--param', nargs=2, action='append',
                       metavar=('parameter', 'value'),
                       help='Override config parameters: --param parameter value')
    args = parser.parse_args()
    sim = setup_simulation(args)
    return run_simulation(sim)

if __name__ == "__main__":
    sys.exit(main())