FROM debian:bullseye

# Install base dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    wget \
    unzip \
    pkg-config \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python3-dev \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /altair-vision/

# workind directory for installing openCv
RUN mkdir openCv-lib
WORKDIR /altair-vision/openCv-lib

# Download OpenCV and opencv_contrib
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip

# Create build directory and configure OpenCV
RUN mkdir -p opencv-4.x/build && cd opencv-4.x/build && \
    cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.x/modules \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_EXAMPLES=OFF \
	  -DBUILD_opencv_python3=ON \
          -DINSTALL_C_EXAMPLES=OFF \
          -DINSTALL_PYTHON_EXAMPLES=OFF \
	  -DPYTHON3_EXECUTABLE=/usr/bin/python3 \
	  -DPYTHON3_INCLUDE_DIR=/usr/include/python3.9 \
	  -DPYTHON3_LIBRARY=/usr/lib/arm-linux-gnueabihf/libpython3.9.so \
	  -DBUILD_TESTS=OFF \
          .. && \
          make -j$(nproc) && make install && ldconfig

# setting workdir for main program
RUN cd .. && mkdir src

# # Build and install OpenCV
# RUN cd opencv-4.x/build && make -j$(nproc) && make install && ldconfig

# Copy your project files
WORKDIR /altair-vision/src
COPY . .

# # Clean and build ball_detection project
# RUN rm -rf ball_detection/build && \
#     mkdir -p ball_detection/build && \
#     cd ball_detection/build && \
#     cmake .. && \
#     make -j$(nproc)

# CMD
CMD ["tail", "-f", "/dev/null"]