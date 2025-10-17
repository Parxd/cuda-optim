# ====== CONFIGURATION ======
NVCC        := nvcc
BUILD_DIR   := build
TARGET_CUDA := $(BUILD_DIR)/sgemm
TARGET_CUTLASS := $(BUILD_DIR)/cutlass_tutorial
SRC_DIR     := csrc/sgemm
CUTLASS_DIR := $(SRC_DIR)/cutlass

# ====== FILES ======
# All source files
ALL_SRC     := $(shell find $(SRC_DIR) -name '*.cc' -o -name '*.cu')

# Files for each target
CUDA_SRC    := $(filter-out $(CUTLASS_DIR)/%, $(ALL_SRC))
CUTLASS_SRC := $(shell find $(CUTLASS_DIR) -name '*.cc' -o -name '*.cu')

# Object files
CUDA_OBJS    := $(patsubst $(SRC_DIR)/%, $(BUILD_DIR)/%, $(CUDA_SRC:.cc=.o))
CUDA_OBJS    := $(CUDA_OBJS:.cu=.o)

CUTLASS_OBJS := $(patsubst $(SRC_DIR)/%, $(BUILD_DIR)/%, $(CUTLASS_SRC:.cc=.o))
CUTLASS_OBJS := $(CUTLASS_OBJS:.cu=.o)

# ====== FLAGS ======
INCLUDES    := -I$(SRC_DIR) \
               -Icutlass/include \
               -Icutlass/tools/util/include

CXXFLAGS    := -std=c++17 -O3 -lineinfo
LDFLAGS     := -lcudart -lcublas

# ====== RULES ======
.PHONY: all cuda cutlass debug clean

all: cuda cutlass

debug: CXXFLAGS := -std=c++17 -O0 -G -g -DDEBUG
debug: clean all

# ----- CUDA Target -----
cuda: $(TARGET_CUDA)

$(TARGET_CUDA): $(CUDA_OBJS)
	@echo "Linking $(TARGET_CUDA)..."
	$(NVCC) $(CUDA_OBJS) -o $@ $(LDFLAGS)

# ----- CUTLASS Target -----
cutlass: $(TARGET_CUTLASS)

$(TARGET_CUTLASS): $(CUTLASS_OBJS)
	@echo "Linking $(TARGET_CUTLASS)..."
	$(NVCC) $(CUTLASS_OBJS) -o $@ $(LDFLAGS)

# ----- Compile Rules -----
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	@echo "Compiling CUDA $<..."
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	@echo "Compiling C++ $<..."
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -x cu -c $< -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET_CUDA) $(TARGET_CUTLASS)
