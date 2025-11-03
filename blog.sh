#!/bin/bash

# LlamaFactory Blog Console Panel
# Author: inksnow
# Purpose: Manage LlamaFactory Blog blog posts, server, and GitHub repository
# Version: 1.0.1
# Date: 2025-11-03

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW_B='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Project Directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Display Help Information
show_help() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}    LlamaFactory Blog Console Panel     ${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${YELLOW_B}Available Commands:${NC}"
    echo ""
    echo -e "  ${GREEN}1${NC} - Start Debug Server"
    echo -e "  ${GREEN}2${NC} - Create New Post"
    echo -e "  ${GREEN}3${NC} - Push to GitHub"
    echo -e "  ${GREEN}4${NC} - Check Server Status"
    echo -e "  ${GREEN}0${NC} - Exit"
    echo ""
    echo -e "${PURPLE}Example: Input 1 to start debug server, input 3 to push to GitHub${NC}"
}

# Check Environment
check_environment() {
    if [ ! -f "hugo.yaml" ]; then
        echo -e "${RED}Error: Please run this script in the project root directory${NC}"
        echo -e "${YELLOW_B}Current directory: $(pwd)${NC}"
        echo -e "${YELLOW_B}Script location: $PROJECT_DIR${NC}"
        exit 1
    fi

    if ! command -v hugo &> /dev/null; then
        echo -e "${RED}Error: Hugo is not installed${NC}"
        echo -e "${YELLOW_B}Please install Hugo: https://gohugo.io/installation/${NC}"
        exit 1
    fi

    if [ ! -d "themes/PaperMod" ]; then
        echo -e "${RED}Error: PaperMod theme not found${NC}"
        echo -e "${YELLOW_B}Please install PaperMod theme: git submodule update --init --recursive${NC}"
        exit 1
    fi

    # Check Git installation
    if ! command -v git &> /dev/null; then
        echo -e "${RED}Error: Git is not installed${NC}"
        exit 1
    fi

    if [ ! -d ".git" ]; then
        echo -e "${RED}Error: Current directory is not a Git repository${NC}"
        echo -e "${YELLOW_B}Please initialize Git repository: git init${NC}"
        exit 1
    fi

    # Check branch
    if [ "$(git branch --show-current)" != "main" ]; then
        echo -e "${RED}Error: Please switch to 'main' branch${NC}"
        echo -e "${YELLOW_B}Current branch: $(git branch --show-current)${NC}"
        echo -e "${YELLOW_B}Please switch to 'main': git checkout main${NC}"
        exit 1
    fi

    # Check remote repository
    if ! git remote get-url origin &> /dev/null; then
        echo -e "${RED}Error: No remote repository configured${NC}"
        echo -e "${YELLOW_B}Please add remote repository: git remote add origin <repository-url>${NC}"
        exit 1
    fi

    # Check Git user configuration
    if [ -z "$(git config user.name)" ] || [ -z "$(git config user.email)" ]; then
        echo -e "${YELLOW_B}Warning: Git user information not configured${NC}"
        echo -e "${YELLOW_B}Please configure: git config --global user.name 'Your Name'${NC}"
        echo -e "${YELLOW_B}Please configure: git config --global user.email 'your.email@example.com'${NC}"
        exit 1
    fi
}

# Start Hugo server
start_server() {
    check_environment

    echo -e "${YELLOW_B}正在启动 Hugo 开发服务器...${NC}"

    # Stop any running server
    pkill -f "hugo server" 2> /dev/null || true
    lsof -ti:1313 | xargs kill -9 2> /dev/null || true
    sleep 2

    echo -e "${GREEN}Server started successfully!${NC}"
    echo -e "${BLUE}Server address: http://localhost:1313${NC}"
    echo -e "${BLUE}Press Ctrl+C to stop the server${NC}"
    echo -e "${BLUE}========================================${NC}"

    hugo server -D --disableFastRender --bind 0.0.0.0 --port 1313
}

# Create new post
new_post() {
    check_environment

    if [ -z "$1" ]; then
        echo -e "${RED}Error: Please specify the post path${NC}"
        echo -e "${YELLOW_B}Usage: $0 new <filename>.md${NC}"
        exit 1
    fi

    echo -e "${YELLOW_B}Creating new post: $1${NC}"
    hugo new "posts/$1"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Post created successfully: content/posts/$1${NC}"
        echo -e "${BLUE}Please edit: content/posts/$1${NC}"
    else
        echo -e "${RED}Post creation failed!${NC}"
        exit 1
    fi
}

# Check server status
check_status() {
    if pgrep -f "hugo server" > /dev/null; then
        echo -e "${GREEN}Hugo server is running${NC}"
        echo -e "${BLUE}Process ID: $(pgrep -f "hugo server")${NC}"
        echo -e "${BLUE}Server address: http://localhost:1313${NC}"
    else
        echo -e "${RED}Hugo server is not running${NC}"
    fi
}

# Deploy site to GitHub Pages
deploy_site() {
    check_environment

    echo -e "${YELLOW_B}Publishing site to GitHub Pages...${NC}"

    # Pull latest changes from remote
    echo -e "${YELLOW_B}Pulling latest changes from remote...${NC}"
    if ! git pull origin main; then
        echo -e "${RED}Pull failed! Please resolve conflicts and try again.${NC}"
        exit 1
    fi

    # Check for uncommitted changes
    if [ -n "$(git status --porcelain)" ]; then
        echo -e "${YELLOW_B}Detected uncommitted changes, committing...${NC}"
        git add .
        git commit -m "[feat] Publish site $(date '+%Y-%m-%d %H:%M:%S')"
    fi

    # Generate random branch name
    RANDOM_BRANCH="tmp-$(date +%s)-$RANDOM"
    echo -e "${YELLOW_B}Creating random branch: ${RANDOM_BRANCH}${NC}"

    # Check current branch status
    echo -e "${BLUE}Current branch: $(git branch --show-current)${NC}"

    # Create and switch to random branch
    echo -e "${YELLOW_B}Creating branch...${NC}"
    if ! git checkout -b "$RANDOM_BRANCH"; then
        echo -e "${RED}Branch creation failed!${NC}"
        echo -e "${YELLOW_B}Attempting to delete existing branch and recreate...${NC}"
        git branch -D "$RANDOM_BRANCH" 2>/dev/null
        git checkout -b "$RANDOM_BRANCH"
    fi

    # Check remote repository configuration
    echo -e "${BLUE}Remote repository: $(git remote -v)${NC}"

    # Push to GitHub
    echo -e "${YELLOW_B}Pushing to branch: ${RANDOM_BRANCH}${NC}"
    if ! git push origin "$RANDOM_BRANCH"; then
        echo -e "${RED}Push failed! Attempting to set upstream branch...${NC}"
        git push --set-upstream origin "$RANDOM_BRANCH"
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Publish successful!${NC}"
        echo -e "${BLUE}Pushed to branch: ${RANDOM_BRANCH}${NC}"
        echo -e "${BLUE}Branch URL: https://github.com/hiyouga/llamafactory-blog/tree/${RANDOM_BRANCH}${NC}"
        echo -e "${YELLOW_B}Publish process may take a few minutes, please check later${NC}"
    else
        echo -e "${RED}Publish failed!${NC}"
        exit 1
    fi

    # Clean up temporary branch
    echo -e "${YELLOW_B}Cleaning up branch: ${RANDOM_BRANCH}${NC}"
    git checkout main && git pull origin main
    git branch -D "$RANDOM_BRANCH" 2>/dev/null
}

# Interactive menu
interactive_menu() {
    while true; do
        show_help
        echo ""
        echo -e "${YELLOW_B}Please enter your choice (0-4): ${NC}"
        read -r choice

        case "$choice" in
            1)
                echo -e "${GREEN}Starting Hugo development server...${NC}"
                start_server
                break
                ;;
            2)
                echo -e "${YELLOW_B}Please enter the file name (e.g., my-post.md): ${NC}"
                read -r post_path
                if [ -n "$post_path" ]; then
                    echo -e "${GREEN}Creating new post: $post_path${NC}"
                    new_post "$post_path"
                else
                    echo -e "${RED}Post path cannot be empty${NC}"
                fi
                echo -e "${GREEN}Press any key to continue...${NC}"
                read -r
                ;;
            3)
                echo -e "${GREEN}Publishing site to GitHub Pages...${NC}"
                deploy_site
                echo -e "${GREEN}Press any key to continue...${NC}"
                read -r
                ;;
            4)
                echo -e "${GREEN}Checking server status...${NC}"
                check_status
                echo -e "${GREEN}Press any key to continue...${NC}"
                read -r
                ;;
            0)
                echo -e "${GREEN}Exiting program...${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option, please enter 0-4${NC}"
                echo -e "${GREEN}Press any key to continue...${NC}"
                read -r
                ;;
        esac

        # Clear screen
        if [ "$choice" != "1" ]; then
            clear
        fi
    done
}

# Main program
if [ $# -eq 0 ]; then
    # No arguments, display interactive menu
    interactive_menu
else
    # With arguments, use command line mode
    case "$1" in
        start)
            start_server
            ;;
        deploy)
            deploy_site
            ;;
        new)
            new_post "$2"
            ;;
        status)
            check_status
            ;;
        help|--help|-h)
            show_help
            ;;
        menu)
            interactive_menu
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo -e "${YELLOW_B}Use $0 menu to display interactive menu${NC}"
            echo -e "${YELLOW_B}Use $0 help to display help information${NC}"
            exit 1
            ;;
    esac
fi
