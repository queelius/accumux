from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps
from conan.tools.files import copy
import os


class AccumuxConan(ConanFile):
    name = "accumux"
    version = "1.0.0"
    description = "Modern C++ library for compositional online data reductions"
    author = "Accumux Project Contributors"
    url = "https://github.com/your-username/accumux"
    license = "MIT"
    topics = ("cpp", "statistics", "numerical", "data-processing", "accumulators")
    
    # Package metadata
    homepage = "https://github.com/your-username/accumux"
    
    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "build_tests": [True, False],
        "enable_coverage": [True, False]
    }
    default_options = {
        "build_tests": False,
        "enable_coverage": False
    }
    
    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "include/*", "tests/*", "examples/*", "LICENSE", "README.md"
    
    # No binary packages - header only
    no_copy_source = False
    
    def requirements(self):
        if self.options.build_tests:
            self.requires("gtest/1.14.0")
    
    def build_requirements(self):
        if self.options.build_tests:
            self.tool_requires("cmake/[>=3.20]")
    
    def layout(self):
        cmake_layout(self)
    
    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        
        tc = CMakeToolchain(self)
        tc.variables["ACCUMUX_BUILD_TESTS"] = self.options.build_tests
        tc.variables["ENABLE_COVERAGE"] = self.options.enable_coverage
        tc.generate()
    
    def build(self):
        cmake = CMake(self)
        cmake.configure()
        if self.options.build_tests:
            cmake.build()
            if not self.conf.get("tools.build:skip_test", default=False):
                cmake.test()
    
    def package(self):
        # Copy license
        copy(self, "LICENSE", src=self.source_folder, dst=os.path.join(self.package_folder, "licenses"))
        
        # Copy headers
        copy(self, "*.hpp", 
             src=os.path.join(self.source_folder, "include"), 
             dst=os.path.join(self.package_folder, "include"))
        
        # Copy CMake files if they exist
        copy(self, "*.cmake", 
             src=os.path.join(self.source_folder, "cmake"),
             dst=os.path.join(self.package_folder, "lib", "cmake", "accumux"),
             keep_path=False)
    
    def package_info(self):
        # Header-only library
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
        
        # Set include directories
        self.cpp_info.includedirs = ["include"]
        
        # Set C++ standard requirement
        self.cpp_info.cppstd = "20"
        
        # Compiler features required
        if self.settings.compiler == "gcc":
            if self.settings.compiler.version < "10":
                raise ConanInvalidConfiguration("GCC 10 or higher required")
        elif self.settings.compiler == "clang":
            if self.settings.compiler.version < "10":
                raise ConanInvalidConfiguration("Clang 10 or higher required")
        elif self.settings.compiler == "Visual Studio":
            if self.settings.compiler.version < "16":
                raise ConanInvalidConfiguration("Visual Studio 2019 (16.0) or higher required")
        
        # Set component name for find_package
        self.cpp_info.set_property("cmake_target_name", "accumux::accumux")
        self.cpp_info.set_property("cmake_file_name", "accumux")
        self.cpp_info.set_property("cmake_target_aliases", ["accumux::accumux"])