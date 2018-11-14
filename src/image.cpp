////////////////////////////////////////////////////////////////////////////////
#include "image.h"
// stb_image for loading textures
#define STB_IMAGE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image.h"
#include <fstream>
////////////////////////////////////////////////////////////////////////////////

bool load_image(const std::string &fname, Image & pixels) {
	int width, height, num_raw_channels;
	unsigned char *data = stbi_load(fname.c_str(), &width, &height, &num_raw_channels, 4);

	// An error is indicated by stbi_load() returning NULL.
	if (data == 0) {
		stbi_image_free(data);
		return false;
	}

	pixels.resize(width, height);
	std::copy(data, data + width * height * sizeof(Pixel),
	reinterpret_cast<unsigned char *> (pixels.data()));
	pixels = pixels.rowwise().reverse().eval();

	stbi_image_free(data);

	return true;
}

std::string load_text(const std::string &fname) {
	std::ifstream file(fname);
	if (file.is_open()) {
		file.seekg(0, file.end);
		const auto size = file.tellg();
		file.seekg(0, file.beg);
		std::string result;
		result.resize(size);
		file.read(&result[0], size);
		return result;
	} else {
		throw std::runtime_error("Could not load file: " + fname);
	}
}