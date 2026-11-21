# Mech Royale bot. Built on linux/amd64; runs with no network, 1 CPU, 512 MB, read-only disk, uid 65534.
FROM python:3.13.1-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# system-wide install (not --user), pinned so the final rebuild is identical
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3
WORKDIR /bot
COPY mechfight.py world.py bot.py /bot/
RUN python3 -m compileall -q /bot && python3 -c "import numpy, scipy.sparse.csgraph, scipy.ndimage; print('deps ok')"
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
