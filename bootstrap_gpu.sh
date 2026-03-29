#!/bin/bash

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Color codes for better output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# Function to check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Function to check command success
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Command '$1' not found"
        exit 1
    fi
}

# Function to wait for apt lock
wait_for_apt() {
    while fuser /var/lib/dpkg/lock >/dev/null 2>&1 || \
          fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || \
          fuser /var/lib/apt/lists/lock >/dev/null 2>&1; do
        log_warn "Waiting for other package operations to complete..."
        sleep 5
    done
}

# Main installation functions
update_system() {
    log_info "Updating system packages..."
    wait_for_apt
    apt update -y || { log_error "Failed to update package lists"; exit 1; }
    apt upgrade -y || { log_error "Failed to upgrade packages"; exit 1; }
}

enable_repos() {
    log_info "Enabling contrib and non-free repositories..."
    
    # Check if repositories are already enabled
    if grep -q "contrib non-free non-free-firmware" /etc/apt/sources.list; then
        log_info "Repositories already configured"
        return
    fi
    
    # Enable all required repository components
    sed -i 's/main/main contrib non-free non-free-firmware/' /etc/apt/sources.list || {
        log_error "Failed to modify sources.list"
        exit 1
    }
    
    apt update -y || { log_error "Failed to update package lists after repo change"; exit 1; }
}

install_nvidia_driver() {
    log_info "Installing NVIDIA driver and essential tools..."
    wait_for_apt
    
    # Install required packages
    local packages=(
        "nvidia-driver"
        "firmware-misc-nonfree"
        "build-essential"
        "linux-headers-$(uname -r)"
        "tmux"
        "git"
        "curl"
        "wget"
        "htop"
        "unzip"
        "ufw"
        "xxd"
        "hexedit"
    )
    
    apt install -y "${packages[@]}" || { 
        log_error "Failed to install packages: ${packages[*]}"
        exit 1
    }
}

install_miniconda() {
    log_info "Installing Miniconda..."
    
    # Check if Miniconda is already installed
    if [[ -d "/opt/miniconda" ]]; then
        log_warn "Miniconda already installed at /opt/miniconda"
        log_info "Using existing Miniconda installation..."
        export PATH="/opt/miniconda/bin:$PATH"
        
        # Initialize conda if not already done
        if [[ -f /opt/miniconda/etc/profile.d/conda.sh ]]; then
            source /opt/miniconda/etc/profile.d/conda.sh
        else
            log_error "Existing Miniconda installation appears corrupted"
            exit 1
        fi
        return
    fi
    
    # Create temporary directory
    local temp_dir=$(mktemp -d)
    cd "$temp_dir" || { log_error "Failed to change to temp directory"; exit 1; }
    
    # Download Miniconda
    if ! wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh; then
        log_error "Failed to download Miniconda"
        rm -rf "$temp_dir"
        exit 1
    fi
    
    # Install Miniconda
    if ! bash miniconda.sh -b -p /opt/miniconda; then
        log_error "Failed to install Miniconda"
        rm -rf "$temp_dir"
        exit 1
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
    
    # Initialize conda for bash
    export PATH="/opt/miniconda/bin:$PATH"
    if command -v conda &>/dev/null; then
        conda init bash >/dev/null 2>&1 || true
    else
        log_error "Conda installation failed"
        exit 1
    fi
}

setup_python_environment() {
    log_info "Setting up Python environment..."
    
    # Source conda environment
    if [[ -f /opt/miniconda/etc/profile.d/conda.sh ]]; then
        source /opt/miniconda/etc/profile.d/conda.sh
    else
        log_error "Conda initialization script not found"
        exit 1
    fi
    
    # Check if environment already exists
    if conda env list | grep -q "^trader "; then
        log_info "Python environment 'trader' already exists"
        return
    fi
    
    # Accept conda Terms of Service if needed
    log_info "Accepting conda Terms of Service..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    
    # Update conda
    conda update -y -n base -c defaults conda >/dev/null 2>&1 || true
    
    # Create Python environment
    log_info "Creating Python 3.10 environment..."
    conda create -y -n trader python=3.10 || { 
        log_error "Failed to create Python environment"
        exit 1
    }
}

install_pytorch() {
    log_info "Installing PyTorch with CUDA support..."
    
    # Activate environment
    if [[ -f /opt/miniconda/etc/profile.d/conda.sh ]]; then
        source /opt/miniconda/etc/profile.d/conda.sh
        conda activate trader >/dev/null 2>&1
    fi
    
    # Check if PyTorch is already installed
    if python -c "import torch" >/dev/null 2>&1 2>&1; then
        log_info "PyTorch already installed"
        return
    fi
    
    # Check NVIDIA driver status
    if ! command -v nvidia-smi &>/dev/null; then
        log_warn "NVIDIA driver not detected, installing CPU-only PyTorch"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || {
            log_error "Failed to install CPU-only PyTorch"
            exit 1
        }
    else
        # Get CUDA version from nvidia-smi
        local cuda_version
        cuda_version=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
        
        if [[ -n "$cuda_version" ]] && [[ "$cuda_version" =~ ^12\.[0-9]+$ ]]; then
            log_info "Installing PyTorch for CUDA $cuda_version"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
                log_warn "CUDA 12.1 installation failed, falling back to CPU version"
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            }
        else
            log_warn "CUDA version not detected or unsupported, installing CPU-only PyTorch"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || {
                log_error "Failed to install PyTorch"
                exit 1
            }
        fi
    fi
}

install_python_packages() {
    log_info "Installing core Python dependencies..."
    
    # Activate environment
    if [[ -f /opt/miniconda/etc/profile.d/conda.sh ]]; then
        source /opt/miniconda/etc/profile.d/conda.sh
        conda activate trader >/dev/null 2>&1
    fi
    
    # Check and install core packages
    local core_packages=("numpy" "pandas" "pyarrow" "fastparquet" "scikit-learn" "matplotlib" "tqdm" "rich" "jupyter" "requests")
    
    # Install packages one by one to handle failures gracefully
    for package in "${core_packages[@]}"; do
        if ! python -c "import $package" >/dev/null 2>&1 2>&1; then
            log_info "Installing $package..."
            pip install "$package" || {
                log_warn "Failed to install $package, continuing with others"
            }
        else
            log_info "$package already installed"
        fi
    done
    
    log_info "Installing NeuralForecast and related dependencies..."
    local ml_packages=("neuralforecast" "einops" "pytorch-lightning" "tensorboard")
    
    for package in "${ml_packages[@]}"; do
        if ! python -c "import $package" >/dev/null 2>&1 2>&1; then
            log_info "Installing $package..."
            pip install "$package" || {
                log_warn "Failed to install $package, continuing with others"
            }
        else
            log_info "$package already installed"
        fi
    done
}

verify_installation() {
    log_info "Verifying installation..."
    
    # Check NVIDIA driver
    if command -v nvidia-smi &>/dev/null; then
        log_info "NVIDIA driver status:"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits || true
    else
        log_warn "NVIDIA driver not detected"
    fi
    
    # Check conda environment
    if [[ -f /opt/miniconda/etc/profile.d/conda.sh ]]; then
        source /opt/miniconda/etc/profile.d/conda.sh
        conda activate trader >/dev/null 2>&1
        if command -v python &>/dev/null; then
            log_info "Python version: $(python --version 2>&1)"
        fi
        
        # Check PyTorch installation
        if python -c "import torch; print(f'PyTorch version: {torch.__version__}')" >/dev/null 2>&1; then
            if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" >/dev/null 2>&1; then
                log_info "PyTorch CUDA status: $(python -c "import torch; print('Available' if torch.cuda.is_available() else 'Not available')")"
            fi
        fi
    fi
}

main() {
    # Check prerequisites
    check_root
    check_command apt
    check_command wget
    
    log_info "Starting Debian 12 RTX4000 Ada deployment setup"
    
    # Execute installation steps
    update_system
    enable_repos
    install_nvidia_driver
    install_miniconda
    setup_python_environment
    install_pytorch
    install_python_packages
    verify_installation
    
    log_info "Deployment setup completed successfully!"
    log_info "To use the Python environment, run: source /opt/miniconda/etc/profile.d/conda.sh && conda activate trader"
    log_info "Reboot the system to ensure NVIDIA drivers are properly loaded: reboot"
}

# Trap to handle script interruption
trap 'log_warn "Script interrupted"; exit 130' INT TERM

# Run main function
main "$@"