#include <vector>
#include <cmath>

#define PI 3.14159265


__global__ void rotate(unsigned char* rgb, unsigned char* g, std::size_t w, std::size_t h, float axeX, float axeY, double theta) {
	auto x1 = blockIdx.x * blockDim.x + threadIdx.x;
	auto y1= blockIdx.y * blockDim.y + threadIdx.y;
	float x0 = axeX; // axe de rotation de l'image en x
	float y0 = axeY; // axe de rotation de l'image en y


	theta = theta * PI / 180.0; // theta est en degré, on le passe donc en radian



	if (x1 < w && y1 < h) {
		int x2 = (int) (cos(theta) * (x1 - x0) + sin(theta) * (y1 - y0));   // nouvelle coordonnée
		int y2 = (int)(-sin(theta) * (x1 - x0) + cos(theta) * (y1 - y0));

		if (x2 >= 0 && x2 < w && y2 >= 0 && y2 < h) {
			g[3 * (y1 * w + x1)] = rgb[3 * (y2 * w + x2)];
			g[3 * (y1 * w + x1) + 1] = rgb[3 * (y2 * w + x2) + 1];
			g[3 * (y1 * w + x1) + 2] = rgb[3 * (y2 * w + x2) + 2];
		}

	}
}


int main()
{
	cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);

	auto rgb = m_in.data; // rgb est l'image en rgb
	auto h = m_in.rows; // la hauteur
	auto w = m_in.cols; // la largeyr

	std::vector< unsigned char > s(w * h * 3); // s est l'image de sortie
	cv::Mat m_out(h, w, CV_8UC3, s.data()); // CV_8UC3 car image en RGB

	unsigned char* rgb_d = nullptr;
	unsigned char* g_d = nullptr;

	cudaMalloc(&rgb_d, 3 * w * h); // *3 car rgb 
	cudaMalloc(&g_d, 3 * w * h); //  *3 car rgb 

	cudaMemcpy(rgb_d, rgb, 3 * w * h, cudaMemcpyHostToDevice); //copie ilage rgb vers device

	dim3 block(32, 32); // a changer pour tester les configurations 32*4 // en général le premier cheiffre c'est 32
	dim3 grid((w - 1) / block.x + 1, (h - 1) / block.y + 1);


	rotate << < grid, block >> > (rgb_d, g_d, w, h, w, h, -180); //  w et h pour faire pivoter par rapport au coins en bas a droite et -180 pour mettre la photo à l'envers


	cudaMemcpy(s.data(), g_d, w * h * 3, cudaMemcpyDeviceToHost); //copie ilage rgb vers device

	//Copie image gris vers host

	cv::imwrite("out.jpg", m_out);

	cudaFree(rgb_d);
	cudaFree(g_d);

	return 0;
}