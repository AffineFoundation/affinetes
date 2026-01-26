FROM python:3.14-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl patch diffutils && \
    rm -rf /var/lib/apt/lists/*



# Python
COPY packages_py.txt /tmp/packages_py.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/packages_py.txt && \
    rm /tmp/packages_py.txt

RUN pip cache purge && \
    rm -rf /root/.cache/pip
    
WORKDIR /sandbox