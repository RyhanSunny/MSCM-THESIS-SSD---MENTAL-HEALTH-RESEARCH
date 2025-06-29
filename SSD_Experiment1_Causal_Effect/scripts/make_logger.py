#!/usr/bin/env python3
"""
make_logger.py - Comprehensive logging wrapper for all make operations

Following CLAUDE.md requirements for version control + timestamps and TDD principles.
Captures full output with timestamps for all make operations.

Usage:
    python scripts/make_logger.py make causal
    python scripts/make_logger.py make all
"""

import subprocess
import sys
import datetime
from pathlib import Path
import argparse
import os


def setup_logging_directory():
    """
    Create logs directory following CLAUDE.md directory structure requirements.
    
    Returns:
    --------
    Path
        Path to logs directory
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def generate_log_filename(make_command: str) -> str:
    """
    Generate timestamped log filename following CLAUDE.md version control requirements.
    
    Parameters:
    -----------
    make_command : str
        The make command being executed (e.g., "make causal")
        
    Returns:
    --------
    str
        Timestamped log filename
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean make command for filename
    clean_command = make_command.replace(" ", "_").replace("make_", "")
    return f"make_{clean_command}_{timestamp}.log"


def run_make_with_logging(make_args: list, log_file: Path) -> int:
    """
    Execute make command with comprehensive logging.
    
    Parameters:
    -----------
    make_args : list
        Make command arguments
    log_file : Path
        Path to log file
        
    Returns:
    --------
    int
        Exit code from make command
    """
    start_time = datetime.datetime.now()
    
    # Prepare environment
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'  # Fix Unicode issues per your requirements
    
    with open(log_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write(f"{'='*80}\n")
        f.write(f"Make Command Log - {start_time.isoformat()}\n")
        f.write(f"Command: {' '.join(make_args)}\n")
        f.write(f"Working Directory: {os.getcwd()}\n")
        f.write(f"Python Version: {sys.version}\n")
        f.write(f"{'='*80}\n\n")
        f.flush()
        
        # Execute command with real-time logging
        try:
            process = subprocess.Popen(
                make_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env
            )
            
            # Stream output to both console and log file
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    timestamped_line = f"[{timestamp}] {output.rstrip()}\n"
                    
                    # Write to log file
                    f.write(timestamped_line)
                    f.flush()
                    
                    # Also print to console (without timestamp to avoid duplication)
                    print(output.rstrip())
                    sys.stdout.flush()
            
            exit_code = process.poll()
            
        except KeyboardInterrupt:
            print("\n⚠️  Make command interrupted by user")
            process.terminate()
            exit_code = 130
        except Exception as e:
            error_msg = f"❌ Error executing make command: {e}\n"
            f.write(error_msg)
            print(error_msg)
            exit_code = 1
        
        # Write footer
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        
        f.write(f"\n{'='*80}\n")
        f.write(f"Command completed: {end_time.isoformat()}\n")
        f.write(f"Duration: {duration}\n")
        f.write(f"Exit code: {exit_code}\n")
        f.write(f"{'='*80}\n")
    
    return exit_code


def main():
    """
    Main function following CLAUDE.md function size requirements (≤50 lines).
    """
    parser = argparse.ArgumentParser(
        description="Comprehensive logging wrapper for make operations"
    )
    parser.add_argument('make_args', nargs='+', 
                       help='Make command and arguments (e.g., make causal)')
    
    args = parser.parse_args()
    
    # Validate that first argument is 'make'
    if args.make_args[0] != 'make':
        print("❌ First argument must be 'make'")
        sys.exit(1)
    
    # Setup logging
    logs_dir = setup_logging_directory()
    make_command = ' '.join(args.make_args)
    log_filename = generate_log_filename(make_command)
    log_file = logs_dir / log_filename
    
    print(f"📝 Logging make operation to: {log_file}")
    print(f"🚀 Executing: {make_command}")
    print(f"{'='*80}")
    
    # Execute with logging
    exit_code = run_make_with_logging(args.make_args, log_file)
    
    print(f"{'='*80}")
    if exit_code == 0:
        print(f"✅ Make command completed successfully")
    else:
        print(f"❌ Make command failed with exit code {exit_code}")
    
    print(f"📄 Full log available at: {log_file}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()