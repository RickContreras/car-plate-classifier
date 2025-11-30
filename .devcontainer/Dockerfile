FROM mcr.microsoft.com/vscode/devcontainers/python:3.12

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]