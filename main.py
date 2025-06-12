#!/usr/bin/env python3
from src.utils.Header import *

def main() -> int:
    parser = argparse.ArgumentParser(description='Robot Simulation')
    parser.add_argument('--config', type=str, default='',
                       help='Name of config file in config directory (leave blank to use default+temporary merge)')
    parser.add_argument('--param', nargs=2, action='append',
                       metavar=('parameter', 'value'),
                       help='Override config parameters: --param parameter value')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Skip cleanup of logs and runs directories')
    parser.add_argument('--cleanup-only', action='store_true',
                       help='Only run cleanup of logs and runs directories, then exit')
    args = parser.parse_args()
    
    # If cleanup-only is requested, run cleanup and exit
    if args.cleanup_only:
        cleanup_logs_and_runs()
        return 0
    else:
        # Cleanup logs and runs directories before starting simulation
        if not args.no_cleanup:
            cleanup_logs_and_runs()
        
        sim = setup_simulation(args)
        return run_simulation(sim)


if __name__ == "__main__":
    sys.exit(main())