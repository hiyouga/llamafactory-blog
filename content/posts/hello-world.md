---
date: '2025-10-20T22:21:47+08:00'
draft: false
title: 'LlamaFactory Blog 项目介绍'
description: '基于 Hugo + PaperMod 的现代化博客系统，支持一键部署到 GitHub Pages'
tags: ['博客', 'Hugo', 'PaperMod', 'GitHub Pages']
categories: ['技术']
---

# LlamaFactory Blog 项目介绍

欢迎来到 LlamaFactory Blog！这是一个基于 Hugo + PaperMod 主题构建的现代化博客系统，支持自动部署到 GitHub Pages。

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/hiyouga/llamafactory-blog.git
cd llamafactory-blog
```

### 2. 拉取子模块（PaperMod 主题）

```bash
git submodule update --init --recursive
```

### 3. 启动项目

```bash
./blog.sh
```

运行后会显示交互式菜单：

```
========================================
    Hugo PaperMod 博客管理脚本
========================================

请选择要执行的操作:

  1 - 启动开发服务器
  2 - 创建新文章
  3 - 发布到 GitHub Pages
  4 - 查看服务器状态
  0 - 退出

示例: 输入 1 启动本地服务器，输入 3 发布网站

请输入选项 (0-4):
```

## 📝 使用流程

### 开发模式

1. **启动本地服务器**：选择 `1`
   - 服务器将在 `http://localhost:1313` 启动
   - 支持热重载，修改内容后自动刷新
   - 按 `Ctrl+C` 停止服务器

2. **创建新文章**：选择 `2`
   - 输入文章路径，如：`posts/我的文章.md`
   - 文章将创建在 `content/posts/` 目录下

3. **编写内容**：编辑 Markdown 文件
   - 支持 Front Matter 配置
   - 支持分类和标签
   - 支持代码高亮和数学公式

### 发布模式

4. **发布到 GitHub Pages**：选择 `3`
   - 自动提交所有更改
   - 推送到 GitHub 仓库
   - GitHub Actions 自动构建并部署


## 🛠️ 技术栈

- **Hugo**: 静态网站生成器
- **PaperMod**: 现代化主题
- **GitHub Pages**: 免费托管
- **GitHub Actions**: 自动部署
- **Markdown**: 内容编写

## ✨ 特性

- 🎨 现代化设计，支持深色/浅色模式
- 📱 响应式布局，移动端友好
- 🔍 内置搜索功能
- 📊 分类和标签系统
- 🚀 一键部署到 GitHub Pages
- ⚡ 极快的加载速度
- 🔧 简单的管理脚本

## 📁 项目结构

```
llamafactory-blog/
├── .github/workflows/    # GitHub Actions 工作流
│   └── hugo.yml         # 自动部署配置
├── content/             # 内容文件
│   ├── posts/           # 博客文章
│   ├── categories/      # 分类页面
│   └── tags/            # 标签页面
├── themes/              # 主题文件
│   └── PaperMod/        # PaperMod 主题
├── hugo.yaml            # Hugo 配置文件
├── blog.sh              # 博客管理脚本
└── README.md            # 项目说明
```

## 🌐 访问地址

- **本地开发**: http://localhost:1313
- **在线访问**: https://hiyouga.github.io/llamafactory-blog/

## 📖 更多功能

- **SEO 优化**: 自动生成 sitemap 和 robots.txt
- **社交分享**: 支持 GitHub、Twitter 等社交平台
- **代码高亮**: 支持多种编程语言
- **数学公式**: 支持 LaTeX 数学公式
- **RSS 订阅**: 自动生成 RSS 源

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

---

*开始你的博客之旅吧！* 🎉
