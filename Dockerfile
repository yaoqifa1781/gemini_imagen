# 1. 基础镜像
FROM python:3.10-slim

# 2. 创建用户 (HF 要求 ID 1000)
RUN useradd -m -u 1000 user

# 3. 切换用户 (后续所有操作都以 user 身份进行)
USER user

# 4. 设置环境变量 (关键！)
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# 5. 设置工作目录
WORKDIR $HOME/app

# 6. 复制并安装依赖
# 注意：使用 --chown=user，这样复制进去的文件直接归 user 所有，不需要 chmod
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. 复制所有代码
COPY --chown=user . .

# 8. 启动
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]