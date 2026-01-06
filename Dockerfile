# Base image with conda
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file first (for better caching)
COPY environment.yml /app/environment.yml

# Create conda environment from YAML
RUN conda env create -f environment.yml && \
    conda clean -afy

# Copy project files
COPY . /app/

# Set environment variables
ENV PATH=/opt/conda/envs/MolDA_CHJ/bin:$PATH
ENV CONDA_DEFAULT_ENV=MolDA_CHJ

# Make conda environment active by default
SHELL ["conda", "run", "-n", "MolDA_CHJ", "/bin/bash", "-c"]

# Default command
CMD ["bash"]
