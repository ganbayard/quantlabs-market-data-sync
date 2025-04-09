import os
import sys
import argparse
import subprocess
from dotenv import load_dotenv

def main():
    # Parse command line arguments - IMPORTANT: Reorder the arguments
    parser = argparse.ArgumentParser(description='Run Alembic migrations for different environments')
    parser.add_argument('--env', choices=['dev', 'prod'], default='dev', help='Target environment (dev or prod)')
    parser.add_argument('command', choices=['init', 'migrate', 'upgrade', 'downgrade', 'revision', 'history', 'current'], 
                        help='Alembic command to run')
    
    # Add all optional arguments after the command positional argument
    parser.add_argument('--autogenerate', action='store_true', help='Autogenerate migrations based on models (for revision command)')
    parser.add_argument('--message', '-m', help='Migration message (for revision command)')
    parser.add_argument('--revision', help='Revision identifier (for upgrade/downgrade commands)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    
    # Parse the known args first, ignoring unknown ones
    args, unknown = parser.parse_known_args()
    
    # Load environment variables
    load_dotenv()
    
    # Set environment variable to indicate which connection to use
    os.environ['MIGRATION_ENV'] = args.env
    
    # Build the base command
    cmd = ['alembic', args.command]
    
    # Handle command-specific arguments
    if args.command == 'revision':
        # Check if --autogenerate is in unknown args (backward compatibility)
        if args.autogenerate or '--autogenerate' in unknown:
            cmd.append('--autogenerate')
        
        if args.message:
            cmd.extend(['-m', args.message])
        else:
            cmd.extend(['-m', f"Migration for {args.env} environment"])
    
    elif args.command in ['upgrade', 'downgrade']:
        if args.revision:
            cmd.append(args.revision)
        else:
            cmd.append('head')
    
    # Add verbose flag if needed
    if args.verbose:
        cmd.append('-v')
    
    # Print info about what we're doing
    connection_type = "LOCAL" if args.env == "dev" else "PRODUCTION"
    print(f"Running '{' '.join(cmd)}' against {connection_type} database")
    
    # Execute the alembic command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Command completed successfully with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print("ERROR: Command failed")
        print(f"Return code: {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()