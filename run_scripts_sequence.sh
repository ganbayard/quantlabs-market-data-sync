#!/bin/bash
# filepath: c:\src\dev\gb\quantlabs-market-data-sync\run_scripts_sequence.sh

# Default environment is dev, can be overridden by command-line parameter
ENV=${1:-dev}

# Telegram bot details (replace with your actual values)
BOT_TOKEN="YOUR_BOT_TOKEN"
CHAT_ID="YOUR_CHAT_ID"

# Create logs directory if it doesn't exist
mkdir -p execution_log

# Log file
LOG_FILE="execution_log/scripts_run_log.txt"

# Function to execute script and log results
run_script() {
    local script_cmd="$1"
    local script_name=$(echo "$script_cmd" | cut -d' ' -f1)
    
    # Log start time
    start_time=$(date +%s)
    start_time_str=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo "[$start_time_str] Starting: $script_cmd --env $ENV" | tee -a "$LOG_FILE"
    
    # Execute the script
    $script_cmd --env $ENV
    exit_code=$?
    
    # Log end time and calculate duration
    end_time=$(date +%s)
    end_time_str=$(date "+%Y-%m-%d %H:%M:%S")
    duration=$((end_time - start_time))
    
    echo "[$end_time_str] Finished: $script_cmd (Status: $exit_code, Duration: ${duration}s)" | tee -a "$LOG_FILE"
    
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
⏱️ *Duration*: ${duration}s
🚦 *Status*: $status (exit code: $exit_code)
------------------"
    
    # Send to Telegram
    curl -s -X POST "https://api.telegram.org/bot$BOT_TOKEN/sendMessage" \
         -d chat_id="$CHAT_ID" \
         -d text="$message" \
         -d parse_mode="Markdown"
    
    # Return exit code to check if we should continue
    return $exit_code
}

# Main script execution
echo "Starting script sequence with environment: $ENV" | tee -a "$LOG_FILE"
echo "===================================================================" | tee -a "$LOG_FILE"

# Array of scripts to run in sequence
scripts=(
    "python scripts/general_info/symbol_fields_update.py"
    "python scripts/etf/holdings_ishare_etf_list.py"
    "python scripts/etf/holdings_ishare_etf_update.py"
    "python scripts/general_info/yf_daily_bar_loader.py --period last_day --workers 1"
    "python scripts/equity2user/equity_technical_indicators.py --period last_day --workers 1"
    "python scripts/sector_rotation/market_breadth_mmtv_update.py --last_day --num_processes 1"
    "python scripts/financial_details/news_collector.py --days 30 --workers 1"
    "python scripts/general_info/company_profile_yfinance.py"
    "python scripts/financial_details/company_financials_yfinance.py --workers 1"
    "python scripts/equity2user/equity2user_history.py --period last_day --workers 1"
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
echo "Script sequence completed at $(date "+%Y-%m-%d %H:%M:%S")" | tee -a "$LOG_FILE"# Add this function to format seconds to HH:MM:SS
format_duration() {
    local seconds=$1
    printf "%02d:%02d:%02d" $((seconds/3600)) $(( (seconds/60) % 60)) $((seconds % 60))
}

# Function to execute script and log results
run_script() {
    local script_cmd="$1"
    local script_name=$(echo "$script_cmd" | cut -d' ' -f1)
    
    # Log start time
    start_time=$(date +%s)
    start_time_str=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo "[$start_time_str] Starting: $script_cmd --env $ENV" | tee -a "$LOG_FILE"
    
    # Execute the script
    $script_cmd --env $ENV
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
    
    # Return exit code to check if we should continue
    return $exit_code
}