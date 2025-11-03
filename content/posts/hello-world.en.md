---
date: '2025-10-20T22:21:47+08:00'
draft: true
title: 'LlamaFactory Blog Introduction'
description: 'A modern blog system based on Hugo + PaperMod theme, supporting one-click deployment to GitHub Pages.'
tags: ['博客', 'Hugo', 'PaperMod', 'GitHub Pages']
categories: ['技术']
---

# LlamaFactory Blog Introduction

Welcome to LlamaFactory Blog! This is a modern blog system built with Hugo + PaperMod theme, supporting one-click deployment to GitHub Pages.

## 🚀 Quick Start

### 1. Clone the Project

```bash
git clone https://github.com/hiyouga/llamafactory-blog.git
cd llamafactory-blog
```

### 2. Pull the Submodule (PaperMod Theme)

```bash
git submodule update --init --recursive
```

### 3. Start the Project

```bash
./blog.sh
```

After running the script, you will see an interactive menu:

```
========================================
     LlamaFactory Blog Control Panel
========================================

Please select the action you want to perform:

  1 - Start the Development Server
  2 - Create a New Post
  3 - Publish to GitHub Pages
  4 - Check Server Status
  0 - Exit

Example: Input 1 to start the local server, input 3 to publish the website

Please enter your choice (0-4):
```

## 📝 Usage

### Development Mode

1. **Start the Development Server**：Select `1`
   - The server will start at `http://localhost:1313`
   - Supports hot reload, content changes will be automatically reflected
   - Press `Ctrl+C` to stop the server

2. **Create a New Post**：Select `2`
   - Input the post path, e.g., `posts/我的文章.md`
   - The post will be created in the `content/posts/` directory

3. **Write Content**：Edit Markdown files
   - Supports Front Matter configuration
   - Supports categories and tags
   - Supports code highlighting and math formulas

### Publication Mode

4. **Publish to GitHub Pages**：Select `3`
   - Automatically commit all changes
   - Push to the GitHub repository
   - GitHub Actions will automatically build and deploy the website


## 🛠️ Reference

- **Hugo**: Static website generator
- **PaperMod**: Modern theme
- **GitHub Pages**: Free hosting
- **GitHub Actions**: Automatic deployment
- **Markdown**: Content writing

## ✨ Features

- 🎨 Modern design, supports dark/light mode
- 📱 Responsive layout, mobile-friendly
- 🔍 Built-in search function
- 📊 Category and tag system
- 🚀 One-click deployment to GitHub Pages
- ⚡ Fast loading speed
- 🔧 Simple management script

## 📁 Project Structure

```
llamafactory-blog/
├── .github/workflows/    # GitHub Actions workflows
│   └── hugo.yml         # Automatic deployment configuration
├── content/             # Content files
│   ├── posts/           # Blog posts
│   ├── categories/      # Category pages
│   └── tags/            # Tag pages
├── themes/              # Theme files
│   └── PaperMod/        # PaperMod theme
├── hugo.yaml            # Hugo configuration file
├── blog.sh              # Blog management script
└── README.md            # Project description
```

## 🌐 Access Address

- **Local Development**: http://localhost:1313
- **Online Access**: https://hiyouga.github.io/llamafactory-blog/

## 📖 More Features

- **SEO Optimization**: Automatically generate sitemap and robots.txt
- **Social Sharing**: Support GitHub, Twitter, etc.
- **Code Highlighting**: Support multiple programming languages
- **Math Formulas**: Support LaTeX math formulas
- **RSS Feed**: Automatically generate RSS feed

## 🤝 Contribution

Welcome to submit issues and pull requests to improve this project!

---

*Start your blog journey now!* 🎉
