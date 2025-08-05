#!/bin/bash

# Amharic H-Net Project Setup Script
# This script sets up the development environment and installs dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    log_info "Detected macOS system"
    PLATFORM="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    log_info "Detected Linux system"
    PLATFORM="linux"
else
    log_warning "Unknown platform: $OSTYPE"
    PLATFORM="unknown"
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        log_info "Found Python $PYTHON_VERSION"
        
        # Check if version is >= 3.9
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            log_success "Python version is compatible"
            return 0
        else
            log_error "Python 3.9+ is required, found $PYTHON_VERSION"
            return 1
        fi
    else
        log_error "Python 3 is not installed"
        return 1
    fi
}

# Function to install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    if [[ "$PLATFORM" == "macos" ]]; then
        if command_exists brew; then
            log_info "Updating Homebrew..."
            brew update
            
            # Install required packages
            log_info "Installing system packages..."
            brew install python@3.11 git curl wget
        else
            log_warning "Homebrew not found. Please install it from https://brew.sh/"
            log_info "Or install Python 3.11+ manually"
        fi
    elif [[ "$PLATFORM" == "linux" ]]; then
        if command_exists apt-get; then
            log_info "Updating package list..."
            sudo apt-get update
            
            log_info "Installing system packages..."
            sudo apt-get install -y python3 python3-pip python3-venv python3-dev \
                build-essential git curl wget libssl-dev libffi-dev
        elif command_exists yum; then
            log_info "Installing system packages with yum..."
            sudo yum install -y python3 python3-pip python3-devel git curl wget \
                gcc gcc-c++ openssl-devel libffi-devel
        else
            log_warning "Package manager not found. Please install Python 3.9+ manually"
        fi
    fi
}

# Function to create virtual environment
setup_venv() {
    log_info "Setting up Python virtual environment..."
    
    if [[ -d "amharic_env" ]]; then
        log_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf amharic_env
        else
            log_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    log_info "Creating virtual environment..."
    python3 -m venv amharic_env
    
    log_info "Activating virtual environment..."
    source amharic_env/bin/activate
    
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment created successfully"
}

# Function to install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # Activate virtual environment
    source amharic_env/bin/activate
    
    # Install core dependencies
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing core dependencies from requirements.txt..."
        pip install -r requirements.txt
    else
        log_warning "requirements.txt not found, installing minimal dependencies..."
        pip install torch transformers numpy pandas tqdm pyyaml
    fi
    
    # Install API dependencies
    if [[ -f "requirements-api.txt" ]]; then
        log_info "Installing API dependencies..."
        pip install -r requirements-api.txt
    fi
    
    # Install development dependencies
    log_info "Installing development dependencies..."
    pip install pytest black flake8 isort mypy jupyter notebook
    
    log_success "Python dependencies installed successfully"
}

# Function to create necessary directories
setup_directories() {
    log_info "Creating project directories..."
    
    directories=(
        "data/raw"
        "data/processed"
        "data/training"
        "data/augmented"
        "models"
        "logs"
        "outputs"
        "backups"
        "configs"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    # Create .gitkeep files for empty directories
    find data -type d -empty -exec touch {}/.gitkeep \;
    
    log_success "Project directories created"
}

# Function to setup configuration files
setup_config() {
    log_info "Setting up configuration files..."
    
    # Create default config if it doesn't exist
    if [[ ! -f "config.yaml" ]]; then
        log_info "Creating default configuration..."
        python3 -c "
import sys
sys.path.insert(0, '.')
from config import ProjectConfig
config = ProjectConfig()
config.to_yaml('config.yaml')
print('Default configuration created')
" 2>/dev/null || log_warning "Could not create default config (config.py may not exist yet)"
    fi
    
    # Create .env file if it doesn't exist
    if [[ ! -f ".env" ]] && [[ -f ".env.example" ]]; then
        log_info "Creating .env file from example..."
        cp .env.example .env
        log_warning "Please edit .env file with your specific settings"
    fi
    
    log_success "Configuration setup complete"
}

# Function to run basic tests
run_tests() {
    log_info "Running basic tests..."
    
    source amharic_env/bin/activate
    
    # Test Python imports
    log_info "Testing Python imports..."
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || log_error "PyTorch import failed"
    python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" || log_error "Transformers import failed"
    python3 -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')" || log_error "FastAPI import failed"
    
    # Test project imports
    if [[ -f "generate.py" ]]; then
        log_info "Testing project modules..."
        python3 -c "from generate import AmharicGenerator; print('AmharicGenerator import successful')" || log_warning "AmharicGenerator import failed"
    fi
    
    log_success "Basic tests completed"
}

# Function to display next steps
show_next_steps() {
    log_success "Setup completed successfully!"
    echo
    log_info "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source amharic_env/bin/activate"
    echo
    echo "2. Start the API server:"
    echo "   python api_server.py"
    echo
    echo "3. Open the web interface:"
    echo "   open web_interface.html"
    echo
    echo "4. Run text generation:"
    echo "   python generate.py --prompt 'ኢትዮጵያ'"
    echo
    echo "5. Run tests:"
    echo "   pytest tests/"
    echo
    log_info "For more information, see README.md"
}

# Main setup function
main() {
    log_info "Starting Amharic H-Net project setup..."
    echo
    
    # Check if we're in the right directory
    if [[ ! -f "README.md" ]] || [[ ! -f "api_server.py" ]]; then
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Run setup steps
    check_python || {
        log_error "Python check failed. Please install Python 3.9+ and try again."
        exit 1
    }
    
    install_system_deps
    setup_venv
    install_python_deps
    setup_directories
    setup_config
    run_tests
    
    echo
    show_next_steps
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-system)
            SKIP_SYSTEM=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --help|-h)
            echo "Amharic H-Net Setup Script"
            echo
            echo "Usage: $0 [options]"
            echo
            echo "Options:"
            echo "  --skip-system    Skip system dependency installation"
            echo "  --skip-tests     Skip running tests"
            echo "  --help, -h       Show this help message"
            echo
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main