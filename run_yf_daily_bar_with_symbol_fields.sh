#!/bin/bash
# filepath: /home/ubuntu/quantlabs-market-data-sync/run_yf_daily_bar_with_symbol_fields.sh
# Default environment is dev, can be overridden by command-line parameter
ENV=${1:-dev}

# Add absolute path configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate the virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Log Python path information for debugging
echo "Using Python: $(which python)" >> "$SCRIPT_DIR/python_path.log"
echo "Python version: $(python --version 2>&1)" >> "$SCRIPT_DIR/python_path.log"

# Bot configuration
BOT_TOKEN="8167802418:AAGlSOFaSYtYueGV0RdAu9AXwNqZDr7XFQQ"
CHAT_ID="1349714573"

# Create logs directory
mkdir -p "$SCRIPT_DIR/execution_log"

EXECUTION_DATE=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="$SCRIPT_DIR/execution_log/scripts_run_${EXECUTION_DATE}.log"

format_duration() {
    local seconds=$1
    printf "%02d:%02d:%02d" $((seconds/3600)) $(( (seconds/60) % 60)) $((seconds % 60))
}

# Function to execute script and log results
run_script() {
    local script_cmd="$1"
    # Extract script path/name without the "python" prefix
    local script_path=$(echo "$script_cmd" | sed 's/python //')
    # Get just the filename without the path for a cleaner display
    local script_name=$(basename "$script_path" | cut -d' ' -f1)
    
    # Log start time
    start_time=$(date +%s)
    start_time_str=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo "[$start_time_str] Starting: $script_cmd --env $ENV" | tee -a "$LOG_FILE"
    
    # Execute the script with the full python path
    eval "$script_cmd --env $ENV"
    exit_code=$?
    
    # Log end time and calculate duration
    end_time=$(date +%s)
    end_time_str=$(date "+%Y-%m-%d %H:%M:%S")
    duration=$((end_time - start_time))
    duration_formatted=$(format_duration $duration)
    
    echo "[$end_time_str] Finished: $script_cmd (Status: $exit_code, Duration: ${duration_formatted})" | tee -a "$LOG_FILE"
    
    # Prepare Telegram notification message
    if [ $exit_code -eq 0 ]; then
        status="✅ SUCCESS"
    else
        status="❌ FAILED"
    fi
    
    message="*Script Execution Report*
------------------
📋 *Script*: \`$script_name\`
🖥️ *Environment*: $ENV
🕒 *Started*: $start_time_str
🕒 *Finished*: $end_time_str
⏱️ *Duration*: ${duration_formatted}
🚦 *Status*: $status (exit code: $exit_code)
------------------"
    
    # Send to Telegram
    curl -s -X POST "https://api.telegram.org/bot$BOT_TOKEN/sendMessage" \
         -d chat_id="$CHAT_ID" \
         -d text="$message" \
         -d parse_mode="Markdown"

    return $exit_code
}

# Log execution start with environment and filename information
echo "Starting script sequence at $(date "+%Y-%m-%d %H:%M:%S")" | tee -a "$LOG_FILE"
echo "Environment: $ENV" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===================================================================" | tee -a "$LOG_FILE"

# Array of scripts to run in sequence
scripts=(
    "python $SCRIPT_DIR/scripts/general_info/symbol_fields_update.py"
    "python $SCRIPT_DIR/scripts/general_info/yf_daily_bar_loader.py --period last_day --workers 1"
)

# Run each script in sequence
for ((i=0; i<${#scripts[@]}; i++)); do
    script="${scripts[$i]}"
    
    # Run the script and get exit code
    run_script "$script"
    script_exit=$?
    
    # Log completion
    echo "------------------------------------------------------------------" | tee -a "$LOG_FILE"
    
    # If it's not the last script, wait 2 minutes before the next one
    if [ $i -lt $((${#scripts[@]}-1)) ]; then
        echo "Waiting 2 minutes before starting next script..." | tee -a "$LOG_FILE"
        sleep 120
    fi
done

echo "===================================================================" | tee -a "$LOG_FILE"
echo "Script sequence completed at $(date "+%Y-%m-%d %H:%M:%S")" | tee -a "$LOG_FILE"