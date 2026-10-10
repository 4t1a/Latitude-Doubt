FROM python:3.12-slim

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Install system utilities and minimal TeX Live tools for PDF building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    texlive-bibtex-extra \
    latexmk \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml /workspace/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt pytest ruff

# Copy entire repository
COPY . /workspace

# Run verification test on container start by default
CMD ["python", "code/verification.py"]
