# Finance Dashboard

A comprehensive Streamlit finance dashboard for stock tracking, drawdown analysis, portfolio management, and financial analysis.

## Features

- Multi-page navigation with sidebar
- User authentication
- Dark/light/system theme modes
- Database connectivity (SQLite and MySQL)
- Stock tracking and analysis
- ETF analysis
- Drawdown analysis
- Holdings analysis
- Risk metrics

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Configuration

- Theme settings are in `.streamlit/config.toml`
- Default credentials: admin / root
- Database settings can be configured in the app

## Project Structure

- `app.py`: Main application entry point
- `.streamlit/`: Streamlit configuration
- `src/`: Source code
  - `database/`: Database models and configuration
  - `pages/`: Individual page modules
  - `components/`: Reusable UI components
  - `utils/`: Utility functions
  - `assets/`: Static assets
- `config/`: Configuration files
- `data/`: Data files (SQLite database)

## License

MIT
