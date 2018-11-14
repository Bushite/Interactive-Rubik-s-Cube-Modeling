// stb_image for loading textures
#include "stb_image.h"
// Linear Algebra Library
#include <Eigen/Dense>

// Type aliases for image textures
typedef Eigen::Matrix<unsigned char, 4, 1> Pixel;
typedef Eigen::Matrix<Pixel, Eigen::Dynamic, Eigen::Dynamic> Image;

// Load an image from a file
bool load_image(const std::string &fname, Image & pixels);

// Load text from a file
std::string load_text(const std::string &fname);