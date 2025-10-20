#!/bin/bash

# LlamaFactory Blog 博客管理脚本
# 作者: inksnow
# 用途: 管理 LlamaFactory Blog 博客的启动、停止、构建等操作

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 项目目录（自动获取当前目录）
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 显示帮助信息
show_help() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}    LlamaFactory Blog 博客管理脚本${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${YELLOW}请选择要执行的操作:${NC}"
    echo ""
    echo -e "  ${GREEN}1${NC} - 启动开发服务器"
    echo -e "  ${GREEN}2${NC} - 创建新文章"
    echo -e "  ${GREEN}3${NC} - 发布到 GitHub Pages"
    echo -e "  ${GREEN}4${NC} - 查看服务器状态"
    echo -e "  ${GREEN}0${NC} - 退出"
    echo ""
    echo -e "${PURPLE}示例: 输入 1 启动本地服务器，输入 3 发布网站${NC}"
}

# 检查环境
check_environment() {
    if [ ! -f "hugo.yaml" ]; then
        echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
    echo -e "${YELLOW}当前目录: $(pwd)${NC}"
    echo -e "${YELLOW}脚本位置: $PROJECT_DIR${NC}"
        exit 1
    fi

    if ! command -v hugo &> /dev/null; then
        echo -e "${RED}错误: Hugo 未安装${NC}"
        echo -e "${YELLOW}请先安装 Hugo: sudo snap install hugo --channel=extended/stable${NC}"
        exit 1
    fi

    if [ ! -d "themes/PaperMod" ]; then
        echo -e "${RED}错误: PaperMod 主题未找到${NC}"
        echo -e "${YELLOW}请确保 themes/PaperMod 目录存在${NC}"
        exit 1
    fi
}

# 启动 Hugo 服务器
start_server() {
    check_environment

    echo -e "${YELLOW}正在启动 Hugo 开发服务器...${NC}"

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
        echo -e "${YELLOW}用法: $0 new posts/文章名.md${NC}"
        exit 1
    fi

    echo -e "${YELLOW}正在创建新文章: $1${NC}"
    hugo new "$1"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}文章创建成功: content/$1${NC}"
        echo -e "${BLUE}请编辑 content/$1 文件${NC}"
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

    echo -e "${YELLOW}正在发布到 GitHub Pages...${NC}"

    # 检查是否有未提交的更改
    if [ -n "$(git status --porcelain)" ]; then
        echo -e "${YELLOW}检测到未提交的更改，正在提交...${NC}"
        git add .
        git commit -m "feat: 更新博客内容 $(date '+%Y-%m-%d %H:%M:%S')"
    fi

    # 推送到 GitHub
    echo -e "${YELLOW}正在推送到 GitHub...${NC}"
    git push origin main

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}发布成功！${NC}"
        echo -e "${BLUE}GitHub Actions 将自动构建并发布到 GitHub Pages${NC}"
    echo -e "${BLUE}访问地址: https://hiyouga.github.io/llamafactory-blog/${NC}"
        echo -e "${YELLOW}发布过程可能需要几分钟，请稍后查看${NC}"
    else
        echo -e "${RED}发布失败！${NC}"
        exit 1
    fi
}

# 交互式菜单
interactive_menu() {
    while true; do
        show_help
        echo ""
        echo -e "${YELLOW}请输入选项 (0-4): ${NC}"
        read -r choice

        case "$choice" in
            1)
                echo -e "${GREEN}启动开发服务器...${NC}"
                start_server
                break
                ;;
            2)
                echo -e "${YELLOW}请输入文章路径 (例如: posts/我的文章.md): ${NC}"
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
            echo -e "${YELLOW}使用 $0 显示交互式菜单${NC}"
            echo -e "${YELLOW}使用 $0 help 显示帮助信息${NC}"
            exit 1
            ;;
    esac
fi
