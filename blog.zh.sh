#!/bin/bash

# LlamaFactory Blog 博客管理脚本
# 作者: inksnow
# 用途: 管理 LlamaFactory Blog 博客的启动、停止、构建等操作
# 版本: 1.0.1
# 日期: 2025-11-03

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW_B='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 项目目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 显示帮助信息
show_help() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}    LlamaFactory Blog 博客管理脚本${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${YELLOW_B}请选择要执行的操作:${NC}"
    echo ""
    echo -e "  ${GREEN}1${NC} - 启动开发服务器"
    echo -e "  ${GREEN}2${NC} - 创建新文章"
    echo -e "  ${GREEN}3${NC} - 推送至 GitHub"
    echo -e "  ${GREEN}4${NC} - 查看服务器状态"
    echo -e "  ${GREEN}0${NC} - 退出"
    echo ""
    echo -e "${PURPLE}示例: 输入 1 启动本地服务器，输入 3 推送至 GitHub${NC}"
}

# 检查环境
check_environment() {
    if [ ! -f "hugo.yaml" ]; then
        echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
        echo -e "${YELLOW_B}当前目录: $(pwd)${NC}"
        echo -e "${YELLOW_B}脚本位置: $PROJECT_DIR${NC}"
        exit 1
    fi

    if ! command -v hugo &> /dev/null; then
        echo -e "${RED}错误: Hugo 未安装${NC}"
        echo -e "${YELLOW_B}请先安装 Hugo: https://gohugo.io/installation/${NC}"
        exit 1
    fi

    if [ ! -d "themes/PaperMod" ]; then
        echo -e "${RED}错误: PaperMod 主题未找到${NC}"
        echo -e "${YELLOW_B}请先安装 PaperMod 主题: git submodule update --init --recursive${NC}"
        exit 1
    fi

    # 检查 git 配置
    if ! command -v git &> /dev/null; then
        echo -e "${RED}错误: Git 未安装${NC}"
        exit 1
    fi

    if [ ! -d ".git" ]; then
        echo -e "${RED}错误: 当前目录不是 Git 仓库${NC}"
        echo -e "${YELLOW_B}请先初始化 Git 仓库: git init${NC}"
        exit 1
    fi

    # 检查分支
    if [ "$(git branch --show-current)" != "main" ]; then
        echo -e "${RED}错误: 请切换到 'main' 分支${NC}"
        echo -e "${YELLOW_B}当前分支: $(git branch --show-current)${NC}"
        echo -e "${YELLOW_B}请切换到 'main': git checkout main${NC}"
        exit 1
    fi

    # 检查远程仓库
    if ! git remote get-url origin &> /dev/null; then
        echo -e "${RED}错误: 未配置远程仓库${NC}"
        echo -e "${YELLOW_B}请添加远程仓库: git remote add origin <仓库URL>${NC}"
        exit 1
    fi

    # 检查 Git 用户配置
    if [ -z "$(git config user.name)" ] || [ -z "$(git config user.email)" ]; then
        echo -e "${YELLOW_B}警告: Git 用户信息未配置${NC}"
        echo -e "${YELLOW_B}请配置: git config --global user.name 'Your Name'${NC}"
        echo -e "${YELLOW_B}请配置: git config --global user.email 'your.email@example.com'${NC}"
        exit 1
    fi
}

# 启动 Hugo 服务器
start_server() {
    check_environment

    echo -e "${YELLOW_B}正在启动 Hugo 开发服务器...${NC}"

    # 停止可能正在运行的服务器
    pkill -f "hugo server" 2>/dev/null || true
    lsof -ti:1313 | xargs kill -9 2>/dev/null || true
    sleep 2

    echo -e "${GREEN}服务器启动中...${NC}"
    echo -e "${BLUE}服务器地址: http://localhost:1313${NC}"
    echo -e "${BLUE}按 Ctrl+C 停止服务器${NC}"
    echo -e "${BLUE}========================================${NC}"

    hugo server -D --disableFastRender --bind 0.0.0.0 --port 1313
}

# 创建新文章
new_post() {
    check_environment

    if [ -z "$1" ]; then
        echo -e "${RED}错误: 请指定文章路径${NC}"
        echo -e "${YELLOW_B}用法: $0 new <文件名>.<语言>.md${NC}"
        exit 1
    fi

    echo -e "${YELLOW_B}正在创建新文章: $1${NC}"
    hugo new "posts/$1"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}文章创建成功: content/posts/$1${NC}"
        echo -e "${BLUE}请编辑: content/posts/$1${NC}"
    else
        echo -e "${RED}文章创建失败！${NC}"
        exit 1
    fi
}

# 查看服务器状态
check_status() {
    if pgrep -f "hugo server" > /dev/null; then
        echo -e "${GREEN}Hugo 服务器正在运行${NC}"
        echo -e "${BLUE}进程 ID: $(pgrep -f "hugo server")${NC}"
        echo -e "${BLUE}访问地址: http://localhost:1313${NC}"
    else
        echo -e "${RED}Hugo 服务器未运行${NC}"
    fi
}

# 发布到 GitHub Pages
deploy_site() {
    check_environment

    echo -e "${YELLOW_B}正在发布到 GitHub Pages...${NC}"

    # 拉取最新更改
    echo -e "${YELLOW_B}正在拉取最新更改...${NC}"
    if ! git pull origin main; then
        echo -e "${RED}拉取失败！请解决冲突并尝试再次发布。${NC}"
        exit 1
    fi

    # 检查是否有未提交的更改
    if [ -n "$(git status --porcelain)" ]; then
        echo -e "${YELLOW_B}检测到未提交的更改，正在提交...${NC}"
        git add .
        git commit -m "[feat] Publish site $(date '+%Y-%m-%d %H:%M:%S')"
    fi

    # 生成随机分支名
    RANDOM_BRANCH="tmp-$(date +%s)-$RANDOM"
    echo -e "${YELLOW_B}创建随机分支: ${RANDOM_BRANCH}${NC}"

    # 检查当前分支状态
    echo -e "${BLUE}当前分支: $(git branch --show-current)${NC}"

    # 创建并切换到随机分支
    echo -e "${YELLOW_B}正在创建分支...${NC}"
    if ! git checkout -b "$RANDOM_BRANCH"; then
        echo -e "${RED}创建分支失败！${NC}"
        echo -e "${YELLOW_B}尝试删除已存在的分支并重新创建...${NC}"
        git branch -D "$RANDOM_BRANCH" 2>/dev/null
        git checkout -b "$RANDOM_BRANCH"
    fi

    # 检查远程仓库配置
    echo -e "${BLUE}远程仓库: $(git remote -v)${NC}"

    # 推送到 GitHub
    echo -e "${YELLOW_B}正在推送到 GitHub 分支: ${RANDOM_BRANCH}${NC}"
    if ! git push origin "$RANDOM_BRANCH"; then
        echo -e "${RED}推送失败！尝试设置上游分支...${NC}"
        git push --set-upstream origin "$RANDOM_BRANCH"
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}发布成功！${NC}"
        echo -e "${BLUE}已推送到分支: ${RANDOM_BRANCH}${NC}"
        echo -e "${BLUE}分支地址: https://github.com/hiyouga/llamafactory-blog/tree/${RANDOM_BRANCH}${NC}"
        echo -e "${YELLOW_B}发布过程可能需要几分钟，请稍后查看${NC}"
    else
        echo -e "${RED}发布失败！${NC}"
        exit 1
    fi

    # 清理临时分支
    echo -e "${YELLOW_B}清理临时分支: ${RANDOM_BRANCH}${NC}"
    git checkout main && git pull origin main
    git branch -D "$RANDOM_BRANCH" 2>/dev/null
}

# 交互式菜单
interactive_menu() {
    while true; do
        show_help
        echo ""
        echo -e "${YELLOW_B}请输入选项 (0-4): ${NC}"
        read -r choice

        case "$choice" in
            1)
                echo -e "${GREEN}启动开发服务器...${NC}"
                start_server
                break
                ;;
            2)
                echo -e "${YELLOW_B}请输入文章路径 (例如: 我的文章.zh.md): ${NC}"
                read -r post_path
                if [ -n "$post_path" ]; then
                    echo -e "${GREEN}创建新文章: $post_path${NC}"
                    new_post "$post_path"
                else
                    echo -e "${RED}文章路径不能为空${NC}"
                fi
                echo -e "${GREEN}按任意键继续...${NC}"
                read -r
                ;;
            3)
                echo -e "${GREEN}发布到 GitHub Pages...${NC}"
                deploy_site
                echo -e "${GREEN}按任意键继续...${NC}"
                read -r
                ;;
            4)
                echo -e "${GREEN}查看服务器状态...${NC}"
                check_status
                echo -e "${GREEN}按任意键继续...${NC}"
                read -r
                ;;
            0)
                echo -e "${GREEN}退出程序${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}无效选项，请输入 0-4${NC}"
                echo -e "${GREEN}按任意键继续...${NC}"
                read -r
                ;;
        esac

        # 清屏（除了启动服务器的选项）
        if [ "$choice" != "1" ]; then
            clear
        fi
    done
}

# 主程序
if [ $# -eq 0 ]; then
    # 没有参数时显示交互式菜单
    interactive_menu
else
    # 有参数时使用命令行模式
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
            echo -e "${RED}未知选项: $1${NC}"
            echo -e "${YELLOW_B}使用 $0 显示交互式菜单${NC}"
            echo -e "${YELLOW_B}使用 $0 help 显示帮助信息${NC}"
            exit 1
            ;;
    esac
fi