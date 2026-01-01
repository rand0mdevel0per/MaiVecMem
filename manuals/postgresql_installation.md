# PostgreSQL 安装教程

本教程将指导您在不同操作系统上安装 PostgreSQL 数据库。

## Windows 系统安装

### 方法一：使用官方安装程序

1. **下载安装程序**
   - 访问 PostgreSQL 官方网站：https://www.postgresql.org/download/windows/
   - 下载适合您系统的版本（推荐下载最新稳定版）

2. **运行安装程序**
   - 双击下载的安装文件
   - 选择安装目录（默认：`C:\Program Files\PostgreSQL\<版本号>`）
   - 选择组件：建议选择所有组件
     - PostgreSQL Server
     - pgAdmin 4（图形化管理工具）
     - Command Line Tools

3. **设置超级用户密码**
   - 为 postgres 用户设置一个强密码
   - 记住这个密码，后续会用到

4. **配置端口**
   - 默认端口：5432
   - 保持默认即可，除非端口已被占用

5. **完成安装**
   - 等待安装完成
   - 取消勾选 "Launch Stack Builder"（可选）
   - 点击 "Finish"

### 方法二：使用 Chocolatey（推荐给开发者）

如果您已经安装了 Chocolatey 包管理器：

```bash
choco install postgresql
```

## macOS 系统安装

### 方法一：使用 Homebrew（推荐）

```bash
# 安装 PostgreSQL
brew install postgresql@15

# 启动 PostgreSQL 服务
brew services start postgresql@15

# 创建数据库
initdb /opt/homebrew/var/postgresql@15
```

### 方法二：使用官方安装程序

1. 访问 https://www.postgresql.org/download/macosx/
2. 下载并运行安装程序
3. 按照向导完成安装

## Linux 系统安装

### Ubuntu/Debian

```bash
# 更新软件包列表
sudo apt update

# 安装 PostgreSQL
sudo apt install postgresql postgresql-contrib

# 启动服务
sudo systemctl start postgresql

# 设置开机自启
sudo systemctl enable postgresql
```

### CentOS/RHEL

```bash
# 安装 PostgreSQL
sudo yum install postgresql-server postgresql-contrib

# 初始化数据库
sudo postgresql-setup initdb

# 启动服务
sudo systemctl start postgresql

# 设置开机自启
sudo systemctl enable postgresql
```

### Fedora

```bash
# 安装 PostgreSQL
sudo dnf install postgresql-server postgresql-contrib

# 初始化数据库
sudo postgresql-setup --initdb

# 启动服务
sudo systemctl start postgresql

# 设置开机自启
sudo systemctl enable postgresql
```

## 验证安装

安装完成后，可以通过以下方式验证：

### Windows

```bash
# 打开命令提示符
psql -U postgres -c "SELECT version();"
```

### macOS/Linux

```bash
# 切换到 postgres 用户
sudo -u postgres psql -c "SELECT version();"
```

如果看到 PostgreSQL 版本信息，说明安装成功！

## 安装后的基本配置

1. **启动 PostgreSQL 服务**
   - Windows: 服务会自动启动
   - macOS: `brew services start postgresql@15`
   - Linux: `sudo systemctl start postgresql`

2. **连接到数据库**
   ```bash
   psql -U postgres
   ```

3. **创建新用户和数据库**（可选但推荐）
   ```sql
   CREATE USER mai_user WITH PASSWORD 'your_password';
   CREATE DATABASE mai_memory OWNER mai_user;
   GRANT ALL PRIVILEGES ON DATABASE mai_memory TO mai_user;
   ```

## 安装 pgAdmin（图形化管理工具）

### Windows
- 安装时已经包含 pgAdmin 4

### macOS
```bash
brew install --cask pgadmin4
```

### Linux
```bash
# Ubuntu/Debian
sudo apt install pgadmin4

# CentOS/RHEL/Fedora
sudo dnf install pgadmin4
```

## 常见问题解决

### 问题1：端口 5432 已被占用
- 解决方案：修改 PostgreSQL 配置文件中的端口号
- 配置文件位置：
  - Windows: `C:\Program Files\PostgreSQL\<版本号>\data\postgresql.conf`
  - macOS/Linux: `/usr/local/var/postgres/postgresql.conf`

### 问题2：无法连接到服务器
- 检查 PostgreSQL 服务是否正在运行
- 检查防火墙设置
- 确认连接参数正确

### 问题3：忘记 postgres 用户密码
- Windows: 使用 `pgAdmin` 重置密码
- macOS/Linux: 使用 `sudo -u postgres psql` 连接后修改

## 下一步

安装完成后，请继续阅读 [PostgreSQL 配置教程](postgresql_configuration.md)。