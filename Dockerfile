FROM continuumio/miniconda3:latest

WORKDIR /app

# Create and activate conda environment
RUN conda create --quiet -n python_env python=3.13 -y
SHELL ["conda", "run", "-n", "python_env", "/bin/bash", "-c"]

# Install poetry and add to PATH
RUN pip install -q poetry
ENV PATH="/root/.local/bin:$PATH"

# Verify poetry installation
RUN poetry --version

# Copy project files
COPY . .

# Install dependencies
RUN poetry -q install --no-interaction --no-ansi

# Set the default command
CMD ["/bin/bash"]
