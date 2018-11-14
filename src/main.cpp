////////////////////////////////////////////////////////////////////////////////
// OpenGL Helpers to reduce the clutter
#include "helpers.h"
#include "image.cpp"
#include <fstream>
// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>
// Linear Algebra Library
#include <Eigen/Dense>
#include <Eigen/Geometry>
// STL headers
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
////////////////////////////////////////////////////////////////////////////////

#define FR 0
#define BA 1
#define RI 2
#define LE 3
#define UP 4
#define DO 5

#define FRAME_NUM 24

// Cube object, with both CPU data (Eigen::Matrix) and GPU data (the VBOs)
struct Cube {
	Eigen::MatrixXf V; // mesh vertices [3 x n]
	Eigen::MatrixXf C; // mesh color [3 x n]
	Eigen::MatrixXf TX; // mesh texture [2 x n]
	Eigen::MatrixXi F; // mesh triangles [3 x m]

	// VBO storing vertex position attributes
	VertexBufferObject V_vbo;

	// VBO storing vertex indices (element buffer)
	VertexBufferObject F_vbo;

	// VBO storing colors
	VertexBufferObject C_vbo;

	// VBO storing textures
	VertexBufferObject T_vbo;

	// VAO storing the layout of the shader program for the object 'bunny'
	VertexArrayObject vao;

	// The transform matrix
	Eigen::MatrixXf T;
};

////////////////////////////////////////////////////////////////////////////////

/*** Global variables declaration ***/

// A 3d array storing all the cubes in the scene
std::vector<Cube> cubes;

// 6 unordered-sets storing the cubes on 6 faces; 
std::unordered_set<int> right_faces;
std::unordered_set<int> left_faces;
std::unordered_set<int> front_faces;
std::unordered_set<int> back_faces;
std::unordered_set<int> up_faces;
std::unordered_set<int> down_faces;

// A vector storing the frames (defferent view matrix) needed to play the animation
std::vector<Eigen::Matrix4f> frames;

// A frame counter to control the animation play
int frame_cnt = -1;

// Save the current time --- it will be used to control the animation
auto t_start = std::chrono::high_resolution_clock::now();

// Rotation option
int rotation_option = -1;

// Rotation options
std::queue<int> rotation_options;
std::queue<int> rotation_started;

std::stack<int> rotation_reversed;

// The id of the selected object
int selected_obj = -1;

// The view matrix
Eigen::MatrixXf view = Eigen::Matrix4f::Identity();

// Projection matrix
Eigen::MatrixXf proj = Eigen::Matrix4f::Identity();

void rotate_front();
void rotate_back();
void rotate_right();
void rotate_left();
void rotate_up();
void rotate_down();
void rotate_front_ccw();
void rotate_back_ccw();
void rotate_right_ccw();
void rotate_left_ccw();
void rotate_up_ccw();
void rotate_down_ccw();
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void key_callback_LEFT_SHIFT(GLFWwindow* window, int key, int scancode, int action, int mods);

////////////////////////////////////////////////////////////////////////////////

// Create a cube and initialize all the parameters
void reset_cubes() {
	cubes.clear();
	right_faces.clear();
	left_faces.clear();
	front_faces.clear();
	back_faces.clear();
	up_faces.clear();
	down_faces.clear();
	frames.clear();
	frame_cnt = -1;
	t_start = std::chrono::high_resolution_clock::now();
	int rotation_option = -1;
	rotation_options = std::queue<int>();
	rotation_started = std::queue<int>();
	rotation_reversed = std::stack<int>();

	// Reset view matrix
	view = Eigen::Affine3f(Eigen::AngleAxis<float>(M_PI/4.0, Eigen::Vector3f::UnitX())).matrix() * 
		Eigen::Affine3f(Eigen::AngleAxis<float>(-M_PI/4.0, Eigen::Vector3f::UnitY())).matrix();
	//Reset projection matrix
	proj << 
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1;

	/*** Create the central cube first ***/
	// Construct 6 faces for the central cube
	Eigen::MatrixXf front(3, 6);
	front << 
		0.15, 0.15, -0.15, 0.15, -0.15, -0.15, 
		0.15, -0.15, -0.15, 0.15, 0.15, -0.15,
		0.15, 0.15, 0.15, 0.15, 0.15, 0.15;

	Eigen::MatrixXf back(3, 6);
	for (int c = 0; c < 6; c++) {
		Eigen::Vector4f v = Eigen::Affine3f(Eigen::AngleAxis<float>(M_PI, Eigen::Vector3f::UnitX())).matrix() * 
								Eigen::Vector4f(front(0, c), front(1, c), front(2, c), 1.0f);
		back.col(c) << v(0), v(1), v(2);
	}

	Eigen::MatrixXf right(3, 6);
	for (int c = 0; c < 6; c++) {
		Eigen::Vector4f v = Eigen::Affine3f(Eigen::AngleAxis<float>(M_PI/2, Eigen::Vector3f::UnitY())).matrix() * 
								Eigen::Vector4f(front(0, c), front(1, c), front(2, c), 1.0f);
		right.col(c) << v(0), v(1), v(2);
	}

	Eigen::MatrixXf left(3, 6);
	for (int c = 0; c < 6; c++) {
		Eigen::Vector4f v = Eigen::Affine3f(Eigen::AngleAxis<float>(-M_PI/2, Eigen::Vector3f::UnitY())).matrix() * 
								Eigen::Vector4f(front(0, c), front(1, c), front(2, c), 1.0f);
		left.col(c) << v(0), v(1), v(2);
	}

	Eigen::MatrixXf up(3, 6);
	for (int c = 0; c < 6; c++) {
		Eigen::Vector4f v = Eigen::Affine3f(Eigen::AngleAxis<float>(-M_PI/2, Eigen::Vector3f::UnitX())).matrix() * 
								Eigen::Vector4f(front(0, c), front(1, c), front(2, c), 1.0f);
		up.col(c) << v(0), v(1), v(2);
	}

	Eigen::MatrixXf down(3, 6);
	for (int c = 0; c < 6; c++) {
		Eigen::Vector4f v = Eigen::Affine3f(Eigen::AngleAxis<float>(M_PI/2, Eigen::Vector3f::UnitX())).matrix() * 
								Eigen::Vector4f(front(0, c), front(1, c), front(2, c), 1.0f);
		down.col(c) << v(0), v(1), v(2);
	}

	// Create the central cube
	Cube center;
	center.V.resize(3, 36);
	center.F.resize(3, 12);
	for (int c = 0; c < 6; c++) {
		center.V.col(c) = front.col(c);
		center.V.col(6 + c)= back.col(c);
		center.V.col(12 + c)= right.col(c);
		center.V.col(18 + c)= left.col(c);
		center.V.col(24 + c)= up.col(c);
		center.V.col(30 + c)= down.col(c);
	}

	for (int c = 0; c < 12; c++) {
		for (int r = 0; r < 3; r++) {
			center.F(r, c) = c * 3 + r;
		}
	}	

	/*** Construct all the cubes from the central cube ***/
	float offset[3] = {-0.30, 0, 0.30};
	for (int x = 0; x < 3; x++) {
		for (int y = 0; y < 3; y++) {
			for (int z = 0; z < 3; z++) {
				Cube cube;
				cube.V.resize(3, 36);
				cube.F.resize(3, 12);
				cube.F = center.F;
				for (int p = 0; p < 36; p++) {
					cube.V.col(p) = center.V.col(p) + Eigen::Vector3f(offset[x], offset[y], offset[z]);
				}

				// Initialize all the cubes to be white
				cube.C = Eigen::MatrixXf::Constant(3, 36, 0.0f);
				// Initilize all the texture
				cube.TX = Eigen::MatrixXf::Constant(2, 36, 0.0f);

				/*** Initialize other parameters ***/
				cube.vao.init();
				cube.vao.bind();

				// Initialize the VBOs
				cube.V_vbo.init(GL_FLOAT, GL_ARRAY_BUFFER);
				cube.C_vbo.init(GL_FLOAT, GL_ARRAY_BUFFER);
				cube.T_vbo.init(GL_FLOAT, GL_ARRAY_BUFFER);
				cube.F_vbo.init(GL_UNSIGNED_INT, GL_ELEMENT_ARRAY_BUFFER);

				// Update the VBOs
				cube.V_vbo.update(cube.V);
				cube.C_vbo.update(cube.C);
				cube.T_vbo.update(cube.TX);
				cube.F_vbo.update(cube.F);

				// Bind the element buffer, this information will be stored in the current VAO
				cube.F_vbo.bind();

				// Unbind the VAO
				cube.vao.unbind();

				// Trandforming matrix
				cube.T = Eigen::Matrix4f::Identity();

				// Add the cube to the array
				cubes.push_back(cube);
			}
		}
	} 

	int f[9] = {2, 5, 8, 11, 14, 17, 20, 23, 26};
	int b[9] = {0, 3, 6, 9, 12, 15, 18, 21, 24};
	int r[9] = {18, 19, 20, 21, 22, 23, 24, 25, 26};
	int l[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	int u[9] = {6, 7, 8, 15, 16, 17, 24, 25, 26};
	int d[9] = {0, 1, 2, 9, 10, 11, 18, 19, 20};
	for (int i = 0; i < 9; i++) {
		front_faces.insert(f[i]);
		back_faces.insert(b[i]);
		right_faces.insert(r[i]);
		left_faces.insert(l[i]);
		up_faces.insert(u[i]);
		down_faces.insert(d[i]);
	}
	
	/*** Initialize colors for all the cube ***/
	// Front faces
	for (const auto &face : front_faces) {
		// 6 vertices on a face of a cube
		for (int v = 0; v < 6; v++) {
			cubes[face].C.col(FR*6+v) << 243/255.0, 243/255.0, 243/255.0; 
		}
		cubes[face].TX.col(FR*6+0) << 1.0/6, 1.0;
		cubes[face].TX.col(FR*6+1) << 1.0/6, 0.5;
		cubes[face].TX.col(FR*6+2) << 0.0, 0.5;
		cubes[face].TX.col(FR*6+3) << 1.0/6, 1.0;
		cubes[face].TX.col(FR*6+4) << 0.0, 1.0;
		cubes[face].TX.col(FR*6+5) << 0.0, 0.5;
	}

	// Back faces
	for (const auto &face : back_faces) {
		// 6 vertices on a face of a cube
		for (int v = 0; v < 6; v++) {
			cubes[face].C.col(BA*6+v) << 240/255.0, 179/255.0, 42/255.0; 
		}
		cubes[face].TX.col(BA*6+0) << 2.0/6, 1.0;
		cubes[face].TX.col(BA*6+1) << 2.0/6, 0.5;
		cubes[face].TX.col(BA*6+2) << 1.0/6, 0.5;
		cubes[face].TX.col(BA*6+3) << 2.0/6, 1.0;
		cubes[face].TX.col(BA*6+4) << 1.0/6, 1.0;
		cubes[face].TX.col(BA*6+5) << 1.0/6, 0.5;
	}

	// Right faces
	for (const auto &face : right_faces) {
		// 6 vertices on a face of a cube
		for (int v = 0; v < 6; v++) {
			cubes[face].C.col(RI*6+v) << 88/255.0, 128/255.0, 243/255.0; 
		}
		cubes[face].TX.col(RI*6+0) << 3.0/6, 1.0;
		cubes[face].TX.col(RI*6+1) << 3.0/6, 0.5;
		cubes[face].TX.col(RI*6+2) << 2.0/6, 0.5;
		cubes[face].TX.col(RI*6+3) << 3.0/6, 1.0;
		cubes[face].TX.col(RI*6+4) << 2.0/6, 1.0;
		cubes[face].TX.col(RI*6+5) << 2.0/6, 0.5;
	}
	
	// Left faces
	for (const auto &face : left_faces) {
		// 6 vertices on a face of a cube
		for (int v = 0; v < 6; v++) {
			cubes[face].C.col(LE*6+v) << 50/255.0, 156/255.0, 88/255.0; 
		}
		cubes[face].TX.col(LE*6+0) << 4.0/6, 1.0;
		cubes[face].TX.col(LE*6+1) << 4.0/6, 0.5;
		cubes[face].TX.col(LE*6+2) << 3.0/6, 0.5;
		cubes[face].TX.col(LE*6+3) << 4.0/6, 1.0;
		cubes[face].TX.col(LE*6+4) << 3.0/6, 1.0;
		cubes[face].TX.col(LE*6+5) << 3.0/6, 0.5;
	}

	// Up faces
	for (const auto &face : up_faces) {
		// 6 vertices on a face of a cube
		for (int v = 0; v < 6; v++) {
			cubes[face].C.col(UP*6+v) << 226/255.0, 112/255.0, 30/255.0; 
		}
		cubes[face].TX.col(UP*6+0) << 5.0/6, 1.0;
		cubes[face].TX.col(UP*6+1) << 5.0/6, 0.5;
		cubes[face].TX.col(UP*6+2) << 4.0/6, 0.5;
		cubes[face].TX.col(UP*6+3) << 5.0/6, 1.0;
		cubes[face].TX.col(UP*6+4) << 4.0/6, 1.0;
		cubes[face].TX.col(UP*6+5) << 4.0/6, 0.5;
	}

	// Down faces
	for (const auto &face : down_faces) {
		// 6 vertices on a face of a cube
		for (int v = 0; v < 6; v++) {
			cubes[face].C.col(DO*6+v) << 221/255.0, 68/255.0, 51/255.0; 
		}
		cubes[face].TX.col(DO*6+0) << 1.0, 1.0;
		cubes[face].TX.col(DO*6+1) << 1.0, 0.5;
		cubes[face].TX.col(DO*6+2) << 5.0/6, 0.5;
		cubes[face].TX.col(DO*6+3) << 1.0, 1.0;
		cubes[face].TX.col(DO*6+4) << 5.0/6, 1.0;
		cubes[face].TX.col(DO*6+5) << 5.0/6, 0.5;
	}

	for (int c = 0; c < 27; c++) {
		cubes[c].C_vbo.update(cubes[c].C);
		cubes[c].T_vbo.update(cubes[c].TX);
	}

	cubes[14].TX.col(FR*6+0) << 1.0/6, 0.5;
	cubes[14].TX.col(FR*6+1) << 1.0/6, 0.0;
	cubes[14].TX.col(FR*6+2) << 0.0, 0.0;
	cubes[14].TX.col(FR*6+3) << 1.0/6, 0.5;
	cubes[14].TX.col(FR*6+4) << 0.0, 0.5;
	cubes[14].TX.col(FR*6+5) << 0.0, 0.0;
	cubes[14].T_vbo.update(cubes[14].TX);
}

////////////////////////////////////////////////////////////////////////////////

// A function to play the animation
void play() {
	if (rotation_option == -1 && rotation_options.size() > 0) {
		rotation_option = rotation_options.front();
		frame_cnt = 0;
	}
	if (frame_cnt != -1 && frame_cnt < FRAME_NUM - 1)  {
		float time = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - t_start).count();
		time *= 24.0; // speed up the animation
		if (time > 1) {
			// If it is the last frame, finish the rotation by setting the model matrix
			// And update the 6 unordered sets
			if (frame_cnt == FRAME_NUM - 2) {
				if (rotation_option == 0) {
					int count = 0;
					for (const auto &face : front_faces) {
						cubes[face].T = frames[count*FRAME_NUM+FRAME_NUM-1];
						count++;

						std::unordered_set<int> temp;
						if (up_faces.find(face) != up_faces.end()) {
							up_faces.erase(face);
							temp.insert(face);
						}
						if (left_faces.find(face) != left_faces.end()) {
							left_faces.erase(face);
							up_faces.insert(face);
						}
						if (down_faces.find(face) != down_faces.end()) {
							down_faces.erase(face);
							left_faces.insert(face);
						}
						if (right_faces.find(face) != right_faces.end()) {
							right_faces.erase(face);
							down_faces.insert(face);
						}
						for (const auto &t : temp) right_faces.insert(t);
					}
				}
				if (rotation_option == 1) {
					int count = 0;
					for (const auto &face : back_faces) {
						cubes[face].T = frames[count*FRAME_NUM+FRAME_NUM-1];
						count++;

						std::unordered_set<int> temp;
						if (up_faces.find(face) != up_faces.end()) {
							up_faces.erase(face);
							temp.insert(face);
						}
						if (right_faces.find(face) != right_faces.end()) {
							right_faces.erase(face);
							up_faces.insert(face);
						}
						
						if (down_faces.find(face) != down_faces.end()) {
							down_faces.erase(face);
							right_faces.insert(face);
						}
						if (left_faces.find(face) != left_faces.end()) {
							left_faces.erase(face);
							down_faces.insert(face);
						}
						for (const auto &t : temp) left_faces.insert(t);
					}
				}
				if (rotation_option == 2) {
					int count = 0;
					for (const auto &face : right_faces) {
						cubes[face].T = frames[count*FRAME_NUM+FRAME_NUM-1];
						count++;

						std::unordered_set<int> temp;
						if (up_faces.find(face) != up_faces.end()) {
							up_faces.erase(face);
							temp.insert(face);
						}
						if (front_faces.find(face) != front_faces.end()) {
							front_faces.erase(face);
							up_faces.insert(face);
						}
						if (down_faces.find(face) != down_faces.end()) {
							down_faces.erase(face);
							front_faces.insert(face);
						}
						if (back_faces.find(face) != back_faces.end()) {
							back_faces.erase(face);
							down_faces.insert(face);
						}
						
						for (const auto &t : temp) back_faces.insert(t);
					}
				}
				if (rotation_option == 3) {
					int count = 0;
					for (const auto &face : left_faces) {
						cubes[face].T = frames[count*FRAME_NUM+FRAME_NUM-1];
						count++;

						std::unordered_set<int> temp;
						if (up_faces.find(face) != up_faces.end()) {
							up_faces.erase(face);
							temp.insert(face);
						}
						if (back_faces.find(face) != back_faces.end()) {
							back_faces.erase(face);
							up_faces.insert(face);
						}
						if (down_faces.find(face) != down_faces.end()) {
							down_faces.erase(face);
							back_faces.insert(face);
						}
						if (front_faces.find(face) != front_faces.end()) {
							front_faces.erase(face);
							down_faces.insert(face);
						}
						
						for (const auto &t : temp) front_faces.insert(t);
					}
				}
				if (rotation_option == 4) {
					int count = 0;
					for (const auto &face : up_faces) {
						cubes[face].T = frames[count*FRAME_NUM+FRAME_NUM-1];
						count++;

						std::unordered_set<int> temp;
						if (front_faces.find(face) != front_faces.end()) {
							front_faces.erase(face);
							temp.insert(face);
						}
						if (right_faces.find(face) != right_faces.end()) {
							right_faces.erase(face);
							front_faces.insert(face);
						}
						if (back_faces.find(face) != back_faces.end()) {
							back_faces.erase(face);
							right_faces.insert(face);
						}
						if (left_faces.find(face) != left_faces.end()) {
							left_faces.erase(face);
							back_faces.insert(face);
						}
						
						for (const auto &t : temp) left_faces.insert(t);
					}
				}
				if (rotation_option == 5) {
					int count = 0;
					for (const auto &face : down_faces) {
						cubes[face].T = frames[count*FRAME_NUM+FRAME_NUM-1];
						count++;

						std::unordered_set<int> temp;
						if (front_faces.find(face) != front_faces.end()) {
							front_faces.erase(face);
							temp.insert(face);
						}
						if (left_faces.find(face) != left_faces.end()) {
							left_faces.erase(face);
							front_faces.insert(face);
						}
						if (back_faces.find(face) != back_faces.end()) {
							back_faces.erase(face);
							left_faces.insert(face);
						}
						if (right_faces.find(face) != right_faces.end()) {
							right_faces.erase(face);
							back_faces.insert(face);
						}
						
						for (const auto &t : temp) right_faces.insert(t);
					}
				}
				if (rotation_option == 6) {
					int count = 0;
					for (const auto &face : front_faces) {
						cubes[face].T = frames[count*FRAME_NUM+FRAME_NUM-1];
						count++;

						std::unordered_set<int> temp;
						if (up_faces.find(face) != up_faces.end()) {
							up_faces.erase(face);
							temp.insert(face);
						}
						if (right_faces.find(face) != right_faces.end()) {
							right_faces.erase(face);
							up_faces.insert(face);
						}
						if (down_faces.find(face) != down_faces.end()) {
							down_faces.erase(face);
							right_faces.insert(face);
						}
						if (left_faces.find(face) != left_faces.end()) {
							left_faces.erase(face);
							down_faces.insert(face);
						}
						for (const auto &t : temp) left_faces.insert(t);
					}
				}
				if (rotation_option == 7) {
					int count = 0;
					for (const auto &face : back_faces) {
						cubes[face].T = frames[count*FRAME_NUM+FRAME_NUM-1];
						count++;

						std::unordered_set<int> temp;
						if (up_faces.find(face) != up_faces.end()) {
							up_faces.erase(face);
							temp.insert(face);
						}
						if (left_faces.find(face) != left_faces.end()) {
							left_faces.erase(face);
							up_faces.insert(face);
						}
						
						if (down_faces.find(face) != down_faces.end()) {
							down_faces.erase(face);
							left_faces.insert(face);
						}
						if (right_faces.find(face) != right_faces.end()) {
							right_faces.erase(face);
							down_faces.insert(face);
						}
						for (const auto &t : temp) right_faces.insert(t);
					}
				}
				if (rotation_option == 8) {
					int count = 0;
					for (const auto &face : right_faces) {
						cubes[face].T = frames[count*FRAME_NUM+FRAME_NUM-1];
						count++;

						std::unordered_set<int> temp;
						if (up_faces.find(face) != up_faces.end()) {
							up_faces.erase(face);
							temp.insert(face);
						}
						if (back_faces.find(face) != back_faces.end()) {
							back_faces.erase(face);
							up_faces.insert(face);
						}
						if (down_faces.find(face) != down_faces.end()) {
							down_faces.erase(face);
							back_faces.insert(face);
						}
						if (front_faces.find(face) != front_faces.end()) {
							front_faces.erase(face);
							down_faces.insert(face);
						}
						
						for (const auto &t : temp) front_faces.insert(t);
					}
				}
				if (rotation_option == 9) {
					int count = 0;
					for (const auto &face : left_faces) {
						cubes[face].T = frames[count*FRAME_NUM+FRAME_NUM-1];
						count++;

						std::unordered_set<int> temp;
						if (up_faces.find(face) != up_faces.end()) {
							up_faces.erase(face);
							temp.insert(face);
						}
						if (front_faces.find(face) != front_faces.end()) {
							front_faces.erase(face);
							up_faces.insert(face);
						}
						if (down_faces.find(face) != down_faces.end()) {
							down_faces.erase(face);
							front_faces.insert(face);
						}
						if (back_faces.find(face) != back_faces.end()) {
							back_faces.erase(face);
							down_faces.insert(face);
						}
						
						for (const auto &t : temp) back_faces.insert(t);
					}
				}
				if (rotation_option == 10) {
					int count = 0;
					for (const auto &face : up_faces) {
						cubes[face].T = frames[count*FRAME_NUM+FRAME_NUM-1];
						count++;

						std::unordered_set<int> temp;
						if (front_faces.find(face) != front_faces.end()) {
							front_faces.erase(face);
							temp.insert(face);
						}
						if (left_faces.find(face) != left_faces.end()) {
							left_faces.erase(face);
							front_faces.insert(face);
						}
						if (back_faces.find(face) != back_faces.end()) {
							back_faces.erase(face);
							left_faces.insert(face);
						}
						if (right_faces.find(face) != right_faces.end()) {
							right_faces.erase(face);
							back_faces.insert(face);
						}
						
						for (const auto &t : temp) right_faces.insert(t);
					}
				}
				if (rotation_option == 11) {
					int count = 0;
					for (const auto &face : down_faces) {
						cubes[face].T = frames[count*FRAME_NUM+FRAME_NUM-1];
						count++;

						std::unordered_set<int> temp;
						if (front_faces.find(face) != front_faces.end()) {
							front_faces.erase(face);
							temp.insert(face);
						}
						if (right_faces.find(face) != right_faces.end()) {
							right_faces.erase(face);
							front_faces.insert(face);
						}
						if (back_faces.find(face) != back_faces.end()) {
							back_faces.erase(face);
							right_faces.insert(face);
						}
						if (left_faces.find(face) != left_faces.end()) {
							left_faces.erase(face);
							back_faces.insert(face);
						}
						
						for (const auto &t : temp) left_faces.insert(t);
					}
				}

				if (!rotation_reversed.empty() && (rotation_reversed.top()+6)%12 == rotation_options.front()) 
					rotation_reversed.pop();
				else 
					rotation_reversed.push(rotation_options.front());
				rotation_options.pop();
				rotation_started.pop();

				std::cout << rotation_options.size() << " steps left" << std::endl;
				if (rotation_options.size() > 0) {
					rotation_option = rotation_options.front();

					frame_cnt = 0;
				}
				else {
					rotation_option = -1;
					frame_cnt = -1;
				}
			}

			t_start = std::chrono::high_resolution_clock::now();
			if (frame_cnt != -1) frame_cnt++;
		}
		else {	
			// Linear interpolate model matrix
			if (rotation_option == 0 || rotation_option == 6) {
				if (rotation_option == 0) rotate_front();
				else rotate_front_ccw();
				int count = 0;
				for (const auto &face : front_faces) {
					cubes[face].T = (1-time) * frames[FRAME_NUM * count + frame_cnt] + time * frames[FRAME_NUM * count + frame_cnt + 1];
					count++;
				}
			}
			if (rotation_option == 1 || rotation_option == 7) {
				if (rotation_option == 1) rotate_back();
				else rotate_back_ccw();
				int count = 0;
				for (const auto &face : back_faces) {
					cubes[face].T = (1-time) * frames[FRAME_NUM * count + frame_cnt] + time * frames[FRAME_NUM * count + frame_cnt + 1];
					count++;
				}
			}
			if (rotation_option == 2 || rotation_option == 8) {
				if (rotation_option == 2) rotate_right();
				else rotate_right_ccw();
				int count = 0;
				for (const auto &face : right_faces) {
					cubes[face].T = (1-time) * frames[FRAME_NUM * count + frame_cnt] + time * frames[FRAME_NUM * count + frame_cnt + 1];
					count++;
				}
			}
			if (rotation_option == 3 || rotation_option == 9) {
				if (rotation_option == 3) rotate_left();
				else rotate_left_ccw();
				int count = 0;
				for (const auto &face : left_faces) {
					cubes[face].T = (1-time) * frames[FRAME_NUM * count + frame_cnt] + time * frames[FRAME_NUM * count + frame_cnt + 1];
					count++;
				}
			}
			if (rotation_option == 4 || rotation_option == 10) {
				if (rotation_option == 4) rotate_up();
				else rotate_up_ccw();
				int count = 0;
				for (const auto &face : up_faces) {
					cubes[face].T = (1-time) * frames[FRAME_NUM * count + frame_cnt] + time * frames[FRAME_NUM * count + frame_cnt + 1];
					count++;
				}
			}
			if (rotation_option == 5 || rotation_option == 11) {
				if (rotation_option == 5) rotate_down();
				else rotate_down_ccw();
				int count = 0;
				for (const auto &face : down_faces) {
					cubes[face].T = (1-time) * frames[FRAME_NUM * count + frame_cnt] + time * frames[FRAME_NUM * count + frame_cnt + 1];
					count++;
				}
			}
		}
	}
}


////////////////////////////////////////////////////////////////////////////////

// Add a rubik's cube to the scene
void key_callback_1(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_RELEASE) {
		reset_cubes();
	}
}

////////////////////////////////////////////////////////////////////////////////

void key_callback_C(GLFWwindow* window, int key, int scancode, int action, int mods) {
	view = Eigen::Affine3f(Eigen::AngleAxis<float>(M_PI/4.0, Eigen::Vector3f::UnitX())).matrix() * 
		Eigen::Affine3f(Eigen::AngleAxis<float>(-M_PI/4.0, Eigen::Vector3f::UnitY())).matrix();
}

////////////////////////////////////////////////////////////////////////////////

void rotate_front() {
	if (rotation_started.front()) return; 

	// Reset the frame buffer
	frames.clear();

	for (const auto &face : front_faces) {
		Eigen::MatrixXf frame = cubes[face].T;
		// 60 frames per rotation
		for (int f = 0; f < FRAME_NUM; f++) {
			frames.push_back(Eigen::Affine3f(Eigen::AngleAxis<float>(-M_PI/2*f/(FRAME_NUM-1), Eigen::Vector3f::UnitZ())).matrix() * cubes[face].T);
		}
	}

	t_start = std::chrono::high_resolution_clock::now();

	frame_cnt = 0;
	rotation_started.front() = true;
}


// Rotate the front face clock wise
void key_callback_F(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_RELEASE) {
		rotation_options.push(0);
		rotation_started.push(false);
	}
}

////////////////////////////////////////////////////////////////////////////////

void rotate_back() {
	if (rotation_started.front()) return; 

	// Reset the frame buffer
	frames.clear();

	for (const auto &face : back_faces) {
		Eigen::MatrixXf frame = cubes[face].T;
		// 60 frames per rotation
		for (int f = 0; f < FRAME_NUM; f++) {
			frames.push_back(Eigen::Affine3f(Eigen::AngleAxis<float>(M_PI/2*f/(FRAME_NUM-1), Eigen::Vector3f::UnitZ())).matrix() * cubes[face].T);
		}
	}

	t_start = std::chrono::high_resolution_clock::now();
	frame_cnt = 0;

	rotation_started.front() = true;
}


// Rotate the back face clock wise
void key_callback_B(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_RELEASE) {
		rotation_options.push(1);
		rotation_started.push(false);
	}
}

////////////////////////////////////////////////////////////////////////////////

void rotate_right() {
	if (rotation_started.front()) return;

	// Reset the frame buffer
	frames.clear();

	for (const auto &face : right_faces) {
		Eigen::MatrixXf frame = cubes[face].T;
		// 60 frames per rotation
		for (int f = 0; f < FRAME_NUM; f++) {
			frames.push_back(Eigen::Affine3f(Eigen::AngleAxis<float>(-M_PI/2*f/(FRAME_NUM-1), Eigen::Vector3f::UnitX())).matrix() * cubes[face].T);
		}
	}
	t_start = std::chrono::high_resolution_clock::now();
	frame_cnt = 0;

	rotation_started.front() = true;
}

// Rotate the right face clock wise
void key_callback_R(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_RELEASE) {
		rotation_options.push(2);
		rotation_started.push(false);
	}
}

////////////////////////////////////////////////////////////////////////////////

void rotate_left() {
	if (rotation_started.front()) return;

	// Reset the frame buffer
	frames.clear();

	for (const auto &face : left_faces) {
		Eigen::MatrixXf frame = cubes[face].T;
		// 60 frames per rotation
		for (int f = 0; f < FRAME_NUM; f++) {
			frames.push_back(Eigen::Affine3f(Eigen::AngleAxis<float>(M_PI/2*f/(FRAME_NUM-1), Eigen::Vector3f::UnitX())).matrix() * cubes[face].T);
		}
	}
	t_start = std::chrono::high_resolution_clock::now();
	frame_cnt = 0;

	rotation_started.front() = true;
}

// Rotate the left face clock wise
void key_callback_L(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_RELEASE) {
		rotation_options.push(3);
		rotation_started.push(false);
	}
}

////////////////////////////////////////////////////////////////////////////////

void rotate_up() {
	if (rotation_started.front()) return;

	// Reset the frame buffer
	frames.clear();

	for (const auto &face : up_faces) {
		Eigen::MatrixXf frame = cubes[face].T;
		// 60 frames per rotation
		for (int f = 0; f < FRAME_NUM; f++) {
			frames.push_back(Eigen::Affine3f(Eigen::AngleAxis<float>(-M_PI/2*f/(FRAME_NUM-1), Eigen::Vector3f::UnitY())).matrix() * cubes[face].T);
		}
	}
	t_start = std::chrono::high_resolution_clock::now();
	frame_cnt = 0;

	rotation_started.front() = true;
}


// Rotate the up face clock wise
void key_callback_U(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_RELEASE) {
		rotation_options.push(4);
		rotation_started.push(false);
	}
}

////////////////////////////////////////////////////////////////////////////////

void rotate_down() {
	if (rotation_started.front()) return;

	// Reset the frame buffer
	frames.clear();

	for (const auto &face : down_faces) {
		Eigen::MatrixXf frame = cubes[face].T;
		// 60 frames per rotation
		for (int f = 0; f < FRAME_NUM; f++) {
			frames.push_back(Eigen::Affine3f(Eigen::AngleAxis<float>(M_PI/2*f/(FRAME_NUM-1), Eigen::Vector3f::UnitY())).matrix() * cubes[face].T);
		}
	}
	t_start = std::chrono::high_resolution_clock::now();
	frame_cnt = 0;

	rotation_started.front() = true;
}

// Rotate the down face clock wise
void key_callback_D(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_RELEASE) {
		rotation_options.push(5);
		rotation_started.push(false);
	}
}

////////////////////////////////////////////////////////////////////////////////

void rotate_front_ccw() {
	if (rotation_started.front()) return; 

	// Reset the frame buffer
	frames.clear();

	for (const auto &face : front_faces) {
		Eigen::MatrixXf frame = cubes[face].T;
		// 60 frames per rotation
		for (int f = 0; f < FRAME_NUM; f++) {
			frames.push_back(Eigen::Affine3f(Eigen::AngleAxis<float>(M_PI/2*f/(FRAME_NUM-1), Eigen::Vector3f::UnitZ())).matrix() * cubes[face].T);
		}
	}

	t_start = std::chrono::high_resolution_clock::now();

	frame_cnt = 0;
	rotation_started.front() = true;
}

////////////////////////////////////////////////////////////////////////////////

void rotate_back_ccw() {
	if (rotation_started.front()) return; 

	// Reset the frame buffer
	frames.clear();

	for (const auto &face : back_faces) {
		Eigen::MatrixXf frame = cubes[face].T;
		// 60 frames per rotation
		for (int f = 0; f < FRAME_NUM; f++) {
			frames.push_back(Eigen::Affine3f(Eigen::AngleAxis<float>(-M_PI/2*f/(FRAME_NUM-1), Eigen::Vector3f::UnitZ())).matrix() * cubes[face].T);
		}
	}

	t_start = std::chrono::high_resolution_clock::now();
	frame_cnt = 0;

	rotation_started.front() = true;
}

////////////////////////////////////////////////////////////////////////////////

void rotate_right_ccw() {
	if (rotation_started.front()) return;

	// Reset the frame buffer
	frames.clear();

	for (const auto &face : right_faces) {
		Eigen::MatrixXf frame = cubes[face].T;
		// 60 frames per rotation
		for (int f = 0; f < FRAME_NUM; f++) {
			frames.push_back(Eigen::Affine3f(Eigen::AngleAxis<float>(M_PI/2*f/(FRAME_NUM-1), Eigen::Vector3f::UnitX())).matrix() * cubes[face].T);
		}
	}
	t_start = std::chrono::high_resolution_clock::now();
	frame_cnt = 0;

	rotation_started.front() = true;
}

////////////////////////////////////////////////////////////////////////////////

void rotate_left_ccw() {
	if (rotation_started.front()) return;

	// Reset the frame buffer
	frames.clear();

	for (const auto &face : left_faces) {
		Eigen::MatrixXf frame = cubes[face].T;
		// 60 frames per rotation
		for (int f = 0; f < FRAME_NUM; f++) {
			frames.push_back(Eigen::Affine3f(Eigen::AngleAxis<float>(-M_PI/2*f/(FRAME_NUM-1), Eigen::Vector3f::UnitX())).matrix() * cubes[face].T);
		}
	}
	t_start = std::chrono::high_resolution_clock::now();
	frame_cnt = 0;

	rotation_started.front() = true;
}

////////////////////////////////////////////////////////////////////////////////

void rotate_up_ccw() {
	if (rotation_started.front()) return;

	// Reset the frame buffer
	frames.clear();

	for (const auto &face : up_faces) {
		Eigen::MatrixXf frame = cubes[face].T;
		// 60 frames per rotation
		for (int f = 0; f < FRAME_NUM; f++) {
			frames.push_back(Eigen::Affine3f(Eigen::AngleAxis<float>(M_PI/2*f/(FRAME_NUM-1), Eigen::Vector3f::UnitY())).matrix() * cubes[face].T);
		}
	}
	t_start = std::chrono::high_resolution_clock::now();
	frame_cnt = 0;

	rotation_started.front() = true;
}

////////////////////////////////////////////////////////////////////////////////

void rotate_down_ccw() {
	if (rotation_started.front()) return;

	// Reset the frame buffer
	frames.clear();

	for (const auto &face : down_faces) {
		Eigen::MatrixXf frame = cubes[face].T;
		// 60 frames per rotation
		for (int f = 0; f < FRAME_NUM; f++) {
			frames.push_back(Eigen::Affine3f(Eigen::AngleAxis<float>(-M_PI/2*f/(FRAME_NUM-1), Eigen::Vector3f::UnitY())).matrix() * cubes[face].T);
		}
	}
	t_start = std::chrono::high_resolution_clock::now();
	frame_cnt = 0;

	rotation_started.front() = true;
}

////////////////////////////////////////////////////////////////////////////////

void key_callback_ccw(GLFWwindow* window, int key, int scancode, int action, int mods) {
	switch (key) {
		case GLFW_KEY_F:
			if (action == GLFW_RELEASE) {
				rotation_options.push(6);
				rotation_started.push(false);
			}
			break;
		case GLFW_KEY_B:
			if (action == GLFW_RELEASE) {
				rotation_options.push(7);
				rotation_started.push(false);
			}
			break;
		case GLFW_KEY_R:
			if (action == GLFW_RELEASE) {
				rotation_options.push(8);
				rotation_started.push(false);
			}
			break;
		case GLFW_KEY_L:
			if (action == GLFW_RELEASE) {
				rotation_options.push(9);
				rotation_started.push(false);
			}
			break;
		case GLFW_KEY_U:
			if (action == GLFW_RELEASE) {
				rotation_options.push(10);
				rotation_started.push(false);
			}
			break;
		case GLFW_KEY_D:
			if (action == GLFW_RELEASE) {
				rotation_options.push(11);
				rotation_started.push(false);
			}
			break;
		case GLFW_KEY_LEFT_SHIFT:
			key_callback_LEFT_SHIFT(window, key, scancode, action, mods);
			break;
		default:
			break;
	}
}

////////////////////////////////////////////////////////////////////////////////

// Holding the left shift will enable counter clock wise roration
void key_callback_LEFT_SHIFT(GLFWwindow* window, int key, int scancode, int action, int mods) {
	// If pressed, enable counter clock wise rotation 
	if (action == GLFW_PRESS) {
		// Register the keyboard callback
		glfwSetKeyCallback(window, key_callback_ccw);
	}

	// If released, resume original rotation;
	if (action == GLFW_RELEASE) {
		// Register the keyboard callback
		glfwSetKeyCallback(window, key_callback);
	}
}

////////////////////////////////////////////////////////////////////////////////

// Solve the cube
void key_callback_SPACE(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_RELEASE) {
		while (!rotation_reversed.empty()) {
			rotation_options.push((rotation_reversed.top()+6)%12);
			rotation_started.push(false);
			rotation_reversed.pop();
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////

double original_xcanonical;
double original_ycanonical;
bool pressed = false;

void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
	if (!pressed) return;
	// Get viewport size (canvas in number of pixels)
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	// Get the size of the window (may be different than the canvas size on retina displays)
	int width_window, height_window;
	glfwGetWindowSize(window, &width_window, &height_window);

	// Deduce position of the mouse in the viewport
	double highdpi = (double) width / (double) width_window;
	xpos *= highdpi;
	ypos *= highdpi;

	// Convert screen position to the canonical viewing volume
	double xcanonical = ((xpos/double(width))*2)-1;
	double ycanonical = (((height-1-ypos)/double(height))*2)-1; // NOTE: y axis is flipped in glfw

	double x = xcanonical - original_xcanonical;
	double y = ycanonical - original_ycanonical;

	view = Eigen::Affine3f(Eigen::AngleAxis<float>(M_PI/2*x, Eigen::Vector3f::UnitY())).matrix() * view;
	view = Eigen::Affine3f(Eigen::AngleAxis<float>(-M_PI/2*y, Eigen::Vector3f::UnitX())).matrix() * view;

	original_xcanonical = xcanonical;
	original_ycanonical = ycanonical;
}

////////////////////////////////////////////////////////////////////////////////

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	// Get viewport size (canvas in number of pixels)
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	// Get the size of the window (may be different than the canvas size on retina displays)
	int width_window, height_window;
	glfwGetWindowSize(window, &width_window, &height_window);

	// Get the position of the mouse in the window
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	// Deduce position of the mouse in the viewport
	double highdpi = (double) width / (double) width_window;
	xpos *= highdpi;
	ypos *= highdpi;

	// Convert screen position to the canonical viewing volume
	double xcanonical = ((xpos/double(width))*2)-1;
	double ycanonical = (((height-1-ypos)/double(height))*2)-1; // NOTE: y axis is flipped in glfw

	if (action == GLFW_PRESS) {
		Eigen::Vector4f ray_canonical;
		ray_canonical << xcanonical, ycanonical, 1.0, 1.0;

		Eigen::Vector3f ray_origin;
		ray_origin << xcanonical, ycanonical, 1.0;
		Eigen::Vector3f ray_direction(0, 0, -1);

		bool intersection_found = false;
		double min_param = INT_MAX;
		for (int m = 0; m < cubes.size(); m++) {
			for (int i = 0; i < cubes[m].F.cols(); i++) {
				Eigen::Vector4f a_temp;
				Eigen::Vector4f b_temp;
				Eigen::Vector4f c_temp;
				a_temp << view * cubes[m].T * Eigen::Vector4f(cubes[m].V(0, cubes[m].F(0, i)), cubes[m].V(1, cubes[m].F(0, i)), cubes[m].V(2, cubes[m].F(0, i)), 1);
				b_temp << view * cubes[m].T * Eigen::Vector4f(cubes[m].V(0, cubes[m].F(1, i)), cubes[m].V(1, cubes[m].F(1, i)), cubes[m].V(2, cubes[m].F(1, i)), 1);
				c_temp << view * cubes[m].T * Eigen::Vector4f(cubes[m].V(0, cubes[m].F(2, i)), cubes[m].V(1, cubes[m].F(2, i)), cubes[m].V(2, cubes[m].F(2, i)), 1);
				Eigen::Vector3f a;
				Eigen::Vector3f b;
				Eigen::Vector3f c;
				a << a_temp(0), a_temp(1), a_temp(2);
				b << b_temp(0), b_temp(1), b_temp(2);
				c << c_temp(0), c_temp(1), c_temp(2);
				
				// Compute intersection
				Eigen::Matrix3f coeff;
				coeff.col(0) = b - a;
				coeff.col(1) = c - a;
				coeff.col(2) = -ray_direction;

				// std::cout << coeff << std::endl;
				Eigen::Vector3f ans = coeff.inverse() * (ray_origin - a);
				if (ans(0) >= 0 && ans(0) <= 1 && ans(1) >= 0 && ans(1) <= 1 && ans(0) + ans(1) >= 0 && ans(0) + ans(1) <= 1) {
					intersection_found = true;
					if (ans(2) < min_param) {
						selected_obj = m;
						min_param = ans(2);
					}
				}
			}
		}
		if (!intersection_found) selected_obj = -1;
	}

	// If no object is selected, then enable track ball
	if (selected_obj == -1) {
		if (action == GLFW_PRESS) {
			glfwSetCursorPosCallback(window, cursor_pos_callback);
			original_xcanonical = xcanonical;
			original_ycanonical = ycanonical;
			pressed = true;
		}
		else 
			pressed = false;
	}
	// If one cube is selected, enable drag and rotate
	else {
		if (action == GLFW_PRESS) {
			original_xcanonical = xcanonical;
			original_ycanonical = ycanonical;
		}
		else {
			if (action == GLFW_RELEASE) {
				double x = xcanonical - original_xcanonical;
				double y = ycanonical - original_ycanonical;

				Eigen::MatrixXf view_original = Eigen::Affine3f(Eigen::AngleAxis<float>(M_PI/4.0, Eigen::Vector3f::UnitX())).matrix() * 
					Eigen::Affine3f(Eigen::AngleAxis<float>(-M_PI/4.0, Eigen::Vector3f::UnitY())).matrix();
				Eigen::MatrixXf view_transform = view * view_original.inverse();

				Eigen::Vector4f xy;
				xy << x, y, 0, 1;

				Eigen::Vector4f xy_temp = view_transform * xy;

				x = xy_temp(0);
				y = xy_temp(1);
				// std::cout << "(" << x << ", " << y << ")" << std::endl;

				if (front_faces.find(selected_obj) != front_faces.end()) {
					if (x * y <= 0 && abs(x) < abs(y)) {
						rotation_options.push(x > 0 ? 0 : 6);
						rotation_started.push(false);
						return;
					}
				}
				if (back_faces.find(selected_obj) != back_faces.end()) {
					if (x * y <= 0 && abs(x) < abs(y)) {
						rotation_options.push(x > 0 ? 7 : 1);
						rotation_started.push(false);
						return;
					}
				}
				if (right_faces.find(selected_obj) != right_faces.end()) {
					if (x * y > 0 && abs(x) < abs(y)) {
						rotation_options.push(x > 0 ? 2 : 8);
						rotation_started.push(false);
						return;
					}
				}
				if (left_faces.find(selected_obj) != left_faces.end()) {
					if (x * y > 0 && abs(x) < abs(y)) {
						rotation_options.push(x > 0 ? 9 : 3);
						rotation_started.push(false);
						return;
					}
				}
				if (up_faces.find(selected_obj) != up_faces.end()) {
					if (abs(x) > abs(y)) {
						rotation_options.push(x > 0 ? 10 : 4);
						rotation_started.push(false);
						return;
					}
				}
				if (down_faces.find(selected_obj) != down_faces.end()) {
					if (abs(x) > abs(y)) {
						rotation_options.push(x > 0 ? 5 : 11);
						rotation_started.push(false);
						return;
					}
				}
			}
		}
	}
	// std::cout << selected_obj << std::endl;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	switch (key) {
		case GLFW_KEY_1:
			key_callback_1(window, key, scancode, action, mods);
			break;
		case GLFW_KEY_C:
			key_callback_C(window, key, scancode, action, mods);
			break;

		case GLFW_KEY_F:
			key_callback_F(window, key, scancode, action, mods);
			break;
		case GLFW_KEY_B:
			key_callback_B(window, key, scancode, action, mods);
			break;
		case GLFW_KEY_R:
			key_callback_R(window, key, scancode, action, mods);
			break;
		case GLFW_KEY_L:
			key_callback_L(window, key, scancode, action, mods);
			break;
		case GLFW_KEY_U:
			key_callback_U(window, key, scancode, action, mods);
			break;
		case GLFW_KEY_D:
			key_callback_D(window, key, scancode, action, mods);
			break;
		case GLFW_KEY_LEFT_SHIFT:
			key_callback_LEFT_SHIFT(window, key, scancode, action, mods);
			break;	
		case GLFW_KEY_SPACE:
			key_callback_SPACE(window, key, scancode, action, mods);
			break;	
		default:
			break;
	}
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
	// Initialize the GLFW library
	if (!glfwInit()) {
		return -1;
	}

	// Activate supersampling
	glfwWindowHint(GLFW_SAMPLES, 8);

	// Ensure that we get at least a 3.2 context
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

	// On apple we have to load a core profile with forward compatibility
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// Create a windowed mode window and its OpenGL context
	GLFWwindow * window = glfwCreateWindow(640, 640, "Interactive Rubik's Cube", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);

	// Load OpenGL and its extensions
	if (!gladLoadGL()) {
		printf("Failed to load OpenGL and its extensions");
		return(-1);
	}
	printf("OpenGL Version %d.%d loaded", GLVersion.major, GLVersion.minor);

	int major, minor, rev;
	major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
	minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
	rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
	printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
	printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
	printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

	// Initialize the OpenGL Program
	// A program controls the OpenGL pipeline and it must contains
	// at least a vertex shader and a fragment shader to be valid
	Program program;
	const GLchar* vertex_shader = R"(
		#version 150 core

		uniform mat4 model;
		uniform mat4 view;
		uniform mat4 proj;

		in vec3 position;
		in vec3 color;
		in vec2 texCoord;

		out vec3 f_color;
		out vec2 f_texCoord;

		void main() {
			gl_Position = proj * view * model * vec4(position, 1.0);

			f_color = color;
			f_texCoord = texCoord;
		}
	)";

	const GLchar* fragment_shader = R"(
		#version 150 core

		uniform vec3 triangle_color;
		uniform sampler2D ourTexture;

		in vec3 f_color;
		in vec2 f_texCoord;

		out vec4 outColor;

		void main() {
			// outColor = vec4(f_color, 1.0);
			// outColor = texture(ourTexture, f_texCoord);
			outColor = texture(ourTexture, f_texCoord) * vec4(f_color, 1.0f);
		}
	)";

	// Compile the two shaders and upload the binary to the GPU
	// Note that we have to explicitly specify that the output "slot" called outColor
	// is the one that we want in the fragment buffer (and thus on screen)
	program.init(vertex_shader, fragment_shader, "outColor");
	program.bind();

	// Load and create a texture 
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture); // All upcoming GL_TEXTURE_2D operations now have effect on this texture object
    // Set border color
    float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    // Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);	// Set texture wrapping to GL_REPEAT (usually basic wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    // Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Load image, create texture and generate mipmaps
    Image pixels;
    if (argc == 1)
    	load_image("../data/stickers.jpg", pixels);
    else if (argc == 2)
    	load_image(argv[1], pixels);
    else 
    	std::cout << "Usage: ./final-project OR ./final-project {JPEG file path}" << std::endl;

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pixels.rows(), pixels.cols(), 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture when done, so we won't accidentily mess up our texture.


	// Register the keyboard callback
	glfwSetKeyCallback(window, key_callback);

	// Register the mouse callback
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	reset_cubes();

	// Loop until the user closes the window
	while (!glfwWindowShouldClose(window)) {
		// Set the size of the viewport (canvas) to the size of the application window (framebuffer)
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		glViewport(0, 0, width, height);
		// Compute the aspect ratio
		float aspect_ratio = float(height)/float(width); // corresponds to the necessary width scaling

		// Clear the framebuffer
		glClearColor(1.0f, 1.0f, 1.0f, 0.5f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Bind texture
		glBindTexture(GL_TEXTURE_2D, texture);

		proj(0, 0) = aspect_ratio;
		// Enable depth test
		glEnable(GL_DEPTH_TEST);
		{
			glUniformMatrix4fv(program.uniform("proj"), 1, GL_FALSE, proj.data());
			glUniformMatrix4fv(program.uniform("view"), 1, GL_FALSE, view.data());


			for (int c = 0; c < cubes.size(); c++) {
				cubes[c].vao.bind();
				program.bindVertexAttribArray("position", cubes[c].V_vbo);
				program.bindVertexAttribArray("color", cubes[c].C_vbo);
				program.bindVertexAttribArray("texCoord", cubes[c].T_vbo);
				 
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, cubes[c].T.data());


				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

				glDrawElements(GL_TRIANGLES, 3 * cubes[c].F.cols(), cubes[c].F_vbo.scalar_type, 0);
				cubes[c].vao.unbind();
			}
			
			// Set the uniform value depending on the time difference
			auto t_now = std::chrono::high_resolution_clock::now();

			// Enable animation play
			play();

			glUniform3f(program.uniform("triangleColor"), 0.0f, 0.0f, 0.0f);
		}

		// Swap front and back buffers
		glfwSwapBuffers(window);

		// Poll for and process events
		glfwPollEvents();
	}

	// Deallocate opengl memory
	program.free();
	for (int c = 0; c < 27; c++) {
		cubes[c].vao.free();
		cubes[c].V_vbo.free();
		cubes[c].C_vbo.free();
		cubes[c].T_vbo.free();
		cubes[c].F_vbo.free();
	}
	// Deallocate glfw internals
	glfwTerminate();
	return 0;
}