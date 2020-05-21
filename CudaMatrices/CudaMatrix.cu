
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaMatrix.h"
#include <iostream>
#include <fstream>

using namespace std;


#define blockD 32


// M = n. righe di A
// N = n. colonne di A = n. righe di B
// H = n. colonne di B
// la matrice C avrà dimensione M x H
template<class T>
void __global__ cudaMatrixMult(const T* A, const T* B, T* C, int M, int N, int H)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	//C[i * other.cols() + j] = -2;

	if (i < M && j < H)
	{
		T sum = 0;
		for (int k = 0; k < N; k++)
		{
			sum += A[i * N + k] * B[k * H + j];
		}

		C[i * H + j] = sum;
	}
}

template<class T>
void __global__  MatrixMulKernel(const T* A, const T* B, T* C, int Width)
{
	// declare cache in the shared memory
	__shared__ T As[blockD][blockD];
	__shared__ T Bs[blockD][blockD];

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	T sum = 0;
	// Loop over the Md and Nd block dimension required to compute the Pd element
	for (int m = 0; m < Width / blockD; m++)
	{
		// collaboratively loading of Md and Nd blocks into shared memory	 
		As[threadIdx.y][threadIdx.x] = A[y * Width + (m * blockD + threadIdx.x)];
		Bs[threadIdx.y][threadIdx.x] = B[(m * blockD + threadIdx.y) * Width + x];
		__syncthreads();

		// keep track of the running sum    
		for (int k = 0; k < blockD; k++)
		{
			sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
		}

		__syncthreads();
	}

	// write back to the global memory
	C[y * Width + x] = sum;
}



template<class T>
void __global__  MatrixMulKernelNonMultipl(const T* A, const T* B, T* C, int M, int N, int H)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (y < M || x < H)
	{
		// declare cache in the shared memory
		__shared__ T As[blockD][blockD];
		__shared__ T Bs[blockD][blockD];
		T sum = 0;

		// Loop sulle sotto-matrici (tiles)
		int N_INTEGER_BLOCKS = floor(((float)N) / blockD);
		for (int m = 0; m < N_INTEGER_BLOCKS; m++)
		{
			if (y < M)
			{
				As[threadIdx.y][threadIdx.x] = A[y * N + (m * blockD + threadIdx.x)];
			}

			if (x < H)
			{
				Bs[threadIdx.y][threadIdx.x] = B[(m * blockD + threadIdx.y) * H + x];
			}

			__syncthreads();

			if (y < M && x < H)
			{
				// keep track of the running sum    
				for (int k = 0; k < blockD; k++)
				{
					sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
				}
			}

			__syncthreads();
		}


		// ultimo blocco
		int DIM_LAST_BLOCK = N % blockD;

		// NB: questo if serve solo per evitare di fare due syncthreads() inutili; se non ci fosse non sarebbe cmq scorretto ma sarebbe meno efficiente
		if (DIM_LAST_BLOCK != 0)
		{
			if (threadIdx.x < DIM_LAST_BLOCK && y < M)
			{
				As[threadIdx.y][threadIdx.x] = A[y * N + (N_INTEGER_BLOCKS * blockD + threadIdx.x)];
			}
			if (threadIdx.y < DIM_LAST_BLOCK && x < H)
			{
				Bs[threadIdx.y][threadIdx.x] = B[(N_INTEGER_BLOCKS * blockD + threadIdx.y) * H + x];
			}

			__syncthreads();

			if (y < M && x < H)
			{
				for (int k = 0; k < DIM_LAST_BLOCK; k++)
				{
					sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
				}
			}

			__syncthreads();
		}

		if (y < M && x < H)
		{
			// write back to the global memory
			C[y * H + x] = sum;
		}
	}
}



template<class T>
void __global__  selfDistShared(const T* A, float* Dist, int M, int N)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (y < M || x < M)
	{
		// declare cache in the shared memory
		__shared__ T As[blockD][blockD];
		__shared__ T Bs[blockD][blockD];
		T sum = 0;

		// Loop sulle sotto-matrici (tiles)
		int N_INTEGER_BLOCKS = floor(((float)N) / blockD);
		for (int m = 0; m < N_INTEGER_BLOCKS; m++)
		{
			if (y < M)
			{
				As[threadIdx.y][threadIdx.x] = A[y * N + (m * blockD + threadIdx.x)];
			}

			if (x < M)
			{
				Bs[threadIdx.y][threadIdx.x] = A[x * N + (m * blockD + threadIdx.y)];
			}

			__syncthreads();

			if (y < M && x < M)
			{
				// keep track of the running sum    
				for (int k = 0; k < blockD; k++)
				{
					sum += (As[threadIdx.y][k] - Bs[k][threadIdx.x]) * (As[threadIdx.y][k] - Bs[k][threadIdx.x]);
				}
			}

			__syncthreads();
		}


		// ultimo blocco
		int DIM_LAST_BLOCK = N % blockD;

		// NB: questo if serve solo per evitare di fare due syncthreads() inutili; se non ci fosse non sarebbe cmq scorretto ma sarebbe meno efficiente
		if (DIM_LAST_BLOCK != 0)
		{
			if (threadIdx.x < DIM_LAST_BLOCK && y < M)
			{
				As[threadIdx.y][threadIdx.x] = A[y * N + (N_INTEGER_BLOCKS * blockD + threadIdx.x)];
			}
			if (threadIdx.y < DIM_LAST_BLOCK && x < M)
			{
				Bs[threadIdx.y][threadIdx.x] = A[x * N + (N_INTEGER_BLOCKS * blockD + threadIdx.y)];
			}

			__syncthreads();

			if (y < M && x < M)
			{
				for (int k = 0; k < DIM_LAST_BLOCK; k++)
				{
					sum += (As[threadIdx.y][k] - Bs[k][threadIdx.x]) * (As[threadIdx.y][k] - Bs[k][threadIdx.x]);
				}
			}

			__syncthreads();
		}

		if (y < M && x < M)
		{
			// write back to the global memory
			Dist[y * M + x] = sqrt((float) sum);
		}
	}
}





// M = n. righe di A = n. righe di B
// N = n. colonne di A = n. colonne di B
// la matrice C avrà dimensione M x N
template <class T>
void __global__ cudaMatrixSum(const T* A, const T* B, T* C, float coefA, float coefB, int M, int N)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < M && j < N)
	{
		C[i * N + j] = coefA * A[i * N + j] + coefB * B[i * N + j];
	}
}


using clock_value_t = long long;

__device__ void sleep(clock_value_t sleep_cycles)
{
	clock_value_t start = clock64();
	clock_value_t cycles_elapsed;
	do { cycles_elapsed = clock64() - start; } while (cycles_elapsed < sleep_cycles);
}


// M = n. righe di A
// N = n. colonne di A
// la matrice Dist avrà dimensione M x M
template <class T>
void __global__ cudaMatrixSelfDist(const T* A, float* Dist, unsigned int* indiciDistMin, int M, int N, unsigned int* numberOfDoneBlocks)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < M && j < M)
	{
		T sum = 0;
		for (int k = 0; k < N; k++)
		{
			sum += (A[i * N + k] - A[j * N + k]) * (A[i * N + k] - A[j * N + k]);
		}

		Dist[i * M + j] = sqrt((float)sum);

		// sincronizzo tutti i thread
		__syncthreads();

		// a questo punto tutti i thread di questo blocco hanno finito; il thread 0 segnala
		// che questo blocco ha terminato, incrementando il contatore globale dei blocchi terminati
		if (threadIdx.x == 0)
		{
			atomicAdd(&(numberOfDoneBlocks[i]), 1);
		}

		// il thread corrispondente alla prima colonna di ogni riga può iniziare a valutare i 3 valori minimi
		// della riga di appartenenza. Lo fa solo quando tutti i blocchi hanno finito
		if (j == 0)
		{
			while (numberOfDoneBlocks[i] < gridDim.x)
			{
				sleep(1);
			}

			float min[3] = { 1.e8, 1.e8, 1.e8 };

			float kMin[3] = { -1, -1, -1 };

			for (int k = 0; k < M; k++)
			{
				float val = Dist[i * M + k];
				if (val < min[0])
				{
					min[2] = min[1];
					min[1] = min[0];
					min[0] = val;

					kMin[2] = kMin[1];
					kMin[1] = kMin[0];
					kMin[0] = k;
				}
				else if (val < min[1])
				{
					min[2] = min[1];
					min[1] = val;

					kMin[2] = kMin[1];
					kMin[1] = k;
				}
				else if (val < min[2])
				{
					min[2] = val;
					kMin[2] = k;
				}
			}
			indiciDistMin[i * 3 + 0] = kMin[0];
			indiciDistMin[i * 3 + 1] = kMin[1];
			indiciDistMin[i * 3 + 2] = kMin[2];
		}
	}
}


// M = n. righe di A
// N = n. colonne di A
// la matrice Dist avrà dimensione M x M
template <class T>
void __global__ cudaMatrixSelfDist2(const T* A, float* Dist, int M, int N)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < M && j < M)
	{
		T sum = 0;
		for (int k = 0; k < N; k++)
		{
			sum += (A[i * N + k] - A[j * N + k]) * (A[i * N + k] - A[j * N + k]);
		}
		Dist[i * M + j] = sqrt((float)sum);
	}
}


template <class T>
void __global__ cudaFindDistBest3(const T* Dist, unsigned int* indiciBest, int M)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < M)
	{
		float min[3] = { 1.e8, 1.e8, 1.e8 };

		float kMin[3] = { -1, -1, -1 };

		for (int k = 0; k < M; k++)
		{
			float val = Dist[i * M + k];
			if (val < min[0])
			{
				min[2] = min[1];
				min[1] = min[0];
				min[0] = val;

				kMin[2] = kMin[1];
				kMin[1] = kMin[0];
				kMin[0] = k;
			}
			else if (val < min[1])
			{
				min[2] = min[1];
				min[1] = val;

				kMin[2] = kMin[1];
				kMin[1] = k;
			}
			else if (val < min[2])
			{
				min[2] = val;
				kMin[2] = k;
			}
		}
		indiciBest[i * 3 + 0] = kMin[0];
		indiciBest[i * 3 + 1] = kMin[1];
		indiciBest[i * 3 + 2] = kMin[2];
	}
}


template <class T>
bool CudaMatrix<T>::sum(const CudaMatrix<T>& other, CudaMatrix<T>& result)
{
	return linearCombination(other, 1, 1, result);
}

template <class T>
bool CudaMatrix<T>::dif(const CudaMatrix<T>& other, CudaMatrix<T>& result)
{
	return linearCombination(other, 1, -1, result);
}

template <class T>
bool CudaMatrix<T>::product(const CudaMatrix<T>& other, CudaMatrix& result)
{
	if (!result.isDataOwned())
	{
		cout << "errore, l'oggetto in cui deve essere salvato il risultato è stato costruito con un buffer di dati esterno. " << endl <<
			"Per evitare leak di risorse l'oggetto in cui salvare il risultato deve essere creato senza dati esterni." << endl;

		return false;
	}

	if (other.Rows() <= 0 || other.Cols() <= 0 || rows <= 0 || cols <= 0)
	{
		cout << "errore, una o più matrici operando sono vuote!" << endl;
		return false;
	}


	if (cols != other.Rows())
	{
		cout << "le matrici non sono compatibili per il prodotto, n. colonne prima matrice: " << cols <<
			", n. righe seconda matrice: " << other.Rows() << endl;
		return false;
	}

	T* d_A;
	T* d_B;
	T* d_C;

	// alloco la memoria su device
	cudaError_t cudaStatus = cudaMalloc((void **)&d_A, sizeof(T) * rows * cols);
	if (cudaStatus != cudaSuccess)
	{
		cout << "fallita l'allocazione sul device della matrice A" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void **)&d_B, sizeof(T) * other.Rows() * other.Cols());
	if (cudaStatus != cudaSuccess)
	{
		cout << "fallita l'allocazione sul device della matrice B" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void **)&d_C, sizeof(T) * rows * other.Cols());
	if (cudaStatus != cudaSuccess)
	{
		cout << "fallita l'allocazione sul device della matrice C" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}

	// copio i dati su device
	cudaStatus = cudaMemcpy((void *)d_A, this->getData(), sizeof(T) * rows * cols, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "fallita la copia della matrice A sul device" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}

	cudaStatus = cudaMemcpy((void *)d_B, other.getData(), sizeof(T) * other.Rows() * other.Cols(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "fallita la copia della matrice B sul device" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}

	// chiamo il kernel
	// ************************************************************************************
	int dimBlocco = 32;
	dim3 threadsPerBlock(dimBlocco, dimBlocco);

	int MAX_DIM_MATRIX = other.Cols() > rows ? other.Cols() : rows;

	int dimGriglia = ceil(float(MAX_DIM_MATRIX) / dimBlocco);

	dim3 blocksPerGrid(dimGriglia, dimGriglia);

	MatrixMulKernelNonMultipl << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, rows, cols, other.Cols());

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		cout << "MatrixMulKernelNonMult ha fallito l'esecuzione" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}

	// copio il risultato dal device all'host
	result.setDimension(rows, other.Cols());

	cudaStatus = cudaMemcpy(result.getData(), d_C, sizeof(T) * rows * other.Cols(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "fallita la copia del risultato dal device all'host" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return true;
}



template <class T>
bool CudaMatrix<T>::productOld(const CudaMatrix<T>& other, CudaMatrix& result)
{
	if (!result.isDataOwned())
	{
		cout << "errore, l'oggetto in cui deve essere salvato il risultato è stato costruito con un buffer di dati esterno. " << endl <<
			"Per evitare leak di risorse l'oggetto in cui salvare il risultato deve essere creato senza dati esterni." << endl;

		return false;
	}

	if (other.Rows() <= 0 || other.Cols() <= 0 || rows <= 0 || cols <= 0)
	{
		cout << "errore, una o più matrici operando sono vuote!" << endl;
		return false;
	}


	if (cols != other.Rows())
	{
		cout << "le matrici non sono compatibili per il prodotto, n. colonne prima matrice: " << cols <<
			", n. righe seconda matrice: " << other.Rows() << endl;
		return false;
	}

	T* d_A;
	T* d_B;
	T* d_C;

	// alloco la memoria su device
	cudaMalloc((void **)&d_A, sizeof(T) * rows * cols);
	cudaMalloc((void **)&d_B, sizeof(T) * other.Rows() * other.Cols());
	cudaMalloc((void **)&d_C, sizeof(T) * rows * other.Cols());

	// copio i dati su device
	cudaMemcpy((void *)d_A, this->getData(), sizeof(T) * rows * cols, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)d_B, other.getData(), sizeof(T) * other.Rows() * other.Cols(), cudaMemcpyHostToDevice);


	// chiamo il kernel
	// ************************************************************************************
	int dimBlocco = 32;
	dim3 threadsPerBlock(dimBlocco, dimBlocco);

	int MAX_DIM_MATRIX = other.Cols() > rows ? other.Cols() : rows;

	int dimGriglia = ceil(float(MAX_DIM_MATRIX) / dimBlocco);

	dim3 blocksPerGrid(dimGriglia, dimGriglia);

	cudaMatrixMult << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, rows, cols, other.Cols());

	// copio il risultato dal device all'host
	result.setDimension(rows, other.Cols());

	cudaMemcpy(result.getData(), d_C, sizeof(T) * rows * other.Cols(), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return true;
}

template <class T>
bool CudaMatrix<T>::productCPU(const CudaMatrix<T>& other, CudaMatrix& result)
{
	if (!result.isDataOwned())
	{
		cout << "errore, l'oggetto in cui deve essere salvato il risultato è stato costruito con un buffer di dati esterno. " << endl <<
			"Per evitare leak di risorse l'oggetto in cui salvare il risultato deve essere creato senza dati esterni." << endl;

		return false;
	}

	if (cols != other.Rows())
	{
		cout << "le matrici non sono compatibili per il prodotto, n. colonne di A: " << cols << ", n. righe di B: " << other.Rows() << endl;
		return false;
	}

	int M = rows;
	int N = cols;
	int H = other.Cols();

	result.setDimension(M, H);

	T* C = result.getData();
	T* B = other.getData();

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < H; j++)
		{
			C[i * H + j] = 0;
			for (int k = 0; k < N; k++)
			{
				C[i * H + j] += data[i * N + k] * B[k * H + j];
			}
		}
	}
	return true;
}

template <class T>
bool CudaMatrix<T>::linearCombination(const CudaMatrix<T>& other, const float alpha, const float beta, CudaMatrix& result)
{
	if (!result.isDataOwned())
	{
		cout << "errore, l'oggetto in cui deve essere salvato il risultato è stato costruito con un buffer di dati esterno. " << endl <<
			"Per evitare leak di risorse l'oggetto in cui salvare il risultato deve essere creato senza dati esterni." << endl;

		return false;
	}

	if (other.Rows() <= 0 || other.Cols() <= 0 || rows <= 0 || cols <= 0)
	{
		cout << "errore, una o più matrici operando sono vuote!" << endl;
		return false;
	}


	if (cols != other.Cols() || rows != other.Rows())
	{
		cout << "le matrici operando non hanno la stessa dimensione!" << endl;
		return false;
	}

	T* d_A;
	T* d_B;
	T* d_C;

	// alloco la memoria su device
	cudaMalloc((void **)&d_A, sizeof(T) * rows * cols);
	cudaMalloc((void **)&d_B, sizeof(T) * rows * cols);
	cudaMalloc((void **)&d_C, sizeof(T) * rows * cols);

	// copio i dati su device
	cudaMemcpy((void *)d_A, this->getData(), sizeof(T) * rows * cols, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)d_B, other.getData(), sizeof(T) * rows * cols, cudaMemcpyHostToDevice);


	// chiamo il kernel
	// ************************************************************************************
	int dimBlocco = 32;
	dim3 threadsPerBlock(dimBlocco, dimBlocco);

	int MAX_DIM_MATRIX = cols > rows ? cols : rows;

	int dimGriglia = ceil(float(MAX_DIM_MATRIX) / dimBlocco);

	dim3 blocksPerGrid(dimGriglia, dimGriglia);

	cudaMatrixSum << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, alpha, beta, rows, cols);


	// copio il risultato dal device all'host

	result.setDimension(rows, cols);

	cudaMemcpy(result.getData(), d_C, sizeof(T) * rows * cols, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return true;
}



template <class T>
bool CudaMatrix<T>::computeSelfDistances(CudaMatrix<float>& Dist, CudaMatrix<unsigned int>& indiciBest)
{
	if (!Dist.isDataOwned() || !indiciBest.isDataOwned())
	{
		cout << "errore, l'oggetto in cui deve essere salvato il risultato è stato costruito con un buffer di dati esterno. " << endl <<
			"Per evitare leak di risorse l'oggetto in cui salvare il risultato deve essere creato senza dati esterni." << endl;

		return false;
	}

	T* d_A;
	float* d_Dist;
	unsigned int* d_indiciBest;
	//unsigned int* d_numberOfDoneBlocks;	// ---------------------------

	// alloco la memoria su device
	cudaError_t cudaStatus = cudaMalloc((void **)&d_A, sizeof(T) * rows * cols);
	if (cudaStatus != cudaSuccess)
	{
		cout << "fallita l'allocazione sul device della matrice A" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void **)&d_Dist, sizeof(float) * rows * rows);
	if (cudaStatus != cudaSuccess)
	{
		cout << "fallita l'allocazione sul device della matrice Dist" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}
	cudaStatus = cudaMalloc((void **)&d_indiciBest, sizeof(unsigned int) * rows * 3);
	if (cudaStatus != cudaSuccess)
	{
		cout << "fallita l'allocazione sul device della matrice indiciBest" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}
	//cudaMalloc((void **)&d_numberOfDoneBlocks, sizeof(unsigned int) * rows);	// -----------------

	// copio i dati su device
	cudaStatus = cudaMemcpy((void *)d_A, this->getData(), sizeof(T) * rows * cols, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "fallita la copia della matrice A sul device" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}
	//cudaMemset((void *)d_numberOfDoneBlocks, 0, sizeof(unsigned int) * rows);	// ------------------

	// chiamo il kernel
	// ************************************************************************************

	int dimBlocco = 32;
	dim3 threadsPerBlock(dimBlocco, dimBlocco);
	int dimGriglia = ceil((float)rows / dimBlocco);
	dim3 blocksPerGrid(dimGriglia, dimGriglia);

	//cudaMatrixSelfDist << <blocksPerGrid, threadsPerBlock >> > (d_A, d_Dist, d_indiciBest, rows, cols, d_numberOfDoneBlocks);	// -------------------
	selfDistShared << <blocksPerGrid, threadsPerBlock >> > (d_A, d_Dist, rows, cols);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		cout << "il kernel Dist ha fallito l'esecuzione" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}

	dim3 threadsPerBlock1d(dimBlocco, 1);
	dim3 blocksPerGrid1d(dimGriglia, 1);

	cudaFindDistBest3 << <blocksPerGrid1d, threadsPerBlock1d >> > (d_Dist, d_indiciBest, rows);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		cout << "il kernel findDist3 ha fallito l'esecuzione" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}


	// copio il risultato dal device all'host
	Dist.setDimension(rows, rows);
	cudaStatus = cudaMemcpy(Dist.getData(), d_Dist, sizeof(float) * rows * rows, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "fallita la copia di Dist dal device all'host" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}

	indiciBest.setDimension(rows, 3);
	cudaStatus = cudaMemcpy(indiciBest.getData(), d_indiciBest, sizeof(unsigned int) * rows * 3, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "fallita la copia di indiciBest dal device all'host" << endl;
		cout << string(cudaGetErrorString(cudaStatus)) << endl;
		return false;
	}


	// ******** debug
	//unsigned int* numberOfDoneBlocks = new unsigned int[rows];
	//cudaMemcpy(numberOfDoneBlocks, d_numberOfDoneBlocks, sizeof(unsigned int) * rows, cudaMemcpyDeviceToHost);
	//delete[] numberOfDoneBlocks;
	// *************


	cudaFree(d_A);
	cudaFree(d_Dist);
	cudaFree(d_indiciBest);
	//cudaFree(d_numberOfDoneBlocks);	// ------------------------

	return true;
}


template <class T>
bool CudaMatrix<T>::computeSelfDistancesOld(CudaMatrix<float>& Dist, CudaMatrix<unsigned int>& indiciBest)
{
	if (!Dist.isDataOwned() || !indiciBest.isDataOwned())
	{
		cout << "errore, l'oggetto in cui deve essere salvato il risultato è stato costruito con un buffer di dati esterno. " << endl <<
			"Per evitare leak di risorse l'oggetto in cui salvare il risultato deve essere creato senza dati esterni." << endl;

		return false;
	}

	T* d_A;
	float* d_Dist;
	unsigned int* d_indiciBest;
	//unsigned int* d_numberOfDoneBlocks;	// ---------------------------

	// alloco la memoria su device
	cudaMalloc((void **)&d_A, sizeof(T) * rows * cols);
	cudaMalloc((void **)&d_Dist, sizeof(float) * rows * rows);
	cudaMalloc((void **)&d_indiciBest, sizeof(unsigned int) * rows * 3);
	//cudaMalloc((void **)&d_numberOfDoneBlocks, sizeof(unsigned int) * rows);	// -----------------

	// copio i dati su device
	cudaMemcpy((void *)d_A, this->getData(), sizeof(T) * rows * cols, cudaMemcpyHostToDevice);
	//cudaMemset((void *)d_numberOfDoneBlocks, 0, sizeof(unsigned int) * rows);	// ------------------

	// chiamo il kernel
	// ************************************************************************************

	int dimBlocco = 32;
	dim3 threadsPerBlock(dimBlocco, dimBlocco);
	int dimGriglia = ceil((float)rows / dimBlocco);
	dim3 blocksPerGrid(dimGriglia, dimGriglia);

	//cudaMatrixSelfDist << <blocksPerGrid, threadsPerBlock >> > (d_A, d_Dist, d_indiciBest, rows, cols, d_numberOfDoneBlocks);	// -------------------
	cudaMatrixSelfDist2 << <blocksPerGrid, threadsPerBlock >> > (d_A, d_Dist, rows, cols);

	dim3 threadsPerBlock1d(dimBlocco, 1);
	dim3 blocksPerGrid1d(dimGriglia, 1);

	cudaFindDistBest3 << <blocksPerGrid1d, threadsPerBlock1d >> > (d_Dist, d_indiciBest, rows);


	// copio il risultato dal device all'host
	Dist.setDimension(rows, rows);
	cudaMemcpy(Dist.getData(), d_Dist, sizeof(float) * rows * rows, cudaMemcpyDeviceToHost);

	indiciBest.setDimension(rows, 3);
	cudaMemcpy(indiciBest.getData(), d_indiciBest, sizeof(unsigned int) * rows * 3, cudaMemcpyDeviceToHost);


	// ******** debug
	//unsigned int* numberOfDoneBlocks = new unsigned int[rows];
	//cudaMemcpy(numberOfDoneBlocks, d_numberOfDoneBlocks, sizeof(unsigned int) * rows, cudaMemcpyDeviceToHost);
	//delete[] numberOfDoneBlocks;
	// *************


	cudaFree(d_A);
	cudaFree(d_Dist);
	cudaFree(d_indiciBest);
	//cudaFree(d_numberOfDoneBlocks);	// ------------------------

	return true;
}



template <class T>
bool CudaMatrix<T>::difCPU(const CudaMatrix<T>& other, CudaMatrix<T>& result)
{
	if (!result.isDataOwned())
	{
		cout << "errore, l'oggetto in cui deve essere salvato il risultato è stato costruito con un buffer di dati esterno. " << endl <<
			"Per evitare leak di risorse l'oggetto in cui salvare il risultato deve essere creato senza dati esterni." << endl;

		return false;
	}


	if (other.Rows() <= 0 || other.Cols() <= 0 || rows <= 0 || cols <= 0)
	{
		cout << "errore, una o più matrici operando sono vuote!" << endl;
		return false;
	}


	if (cols != other.Cols() || rows != other.Rows())
	{
		cout << "le matrici operando non hanno la stessa dimensione!" << endl;
		return false;
	}

	result.setDimension(rows, cols);

	T* dataOther = other.getData();
	T* dataResult = result.getData();

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			//dataResult[i * cols + j] = data[i * cols + j] - dataOther[i * cols + j]; // abs((float)();
			result(i, j) = data[i * cols + j] - dataOther[i * cols + j];
		}
	}


	return true;
}



template <class T>
bool CudaMatrix<T>::computeSelfDistancesCPU(CudaMatrix<float>& Dist, CudaMatrix<unsigned int>& indiciBest)
{
	if (!Dist.isDataOwned() || !indiciBest.isDataOwned())
	{
		cout << "errore, l'oggetto in cui deve essere salvato il risultato è stato costruito con un buffer di dati esterno. " << endl <<
			"Per evitare leak di risorse l'oggetto in cui salvare il risultato deve essere creato senza dati esterni." << endl;

		return false;
	}

	Dist.setDimension(rows, rows);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			Dist(i, j) = 0;
			for (int k = 0; k < cols; k++)
			{
				Dist(i, j) += (this->operator()(i, k) - this->operator()(j, k)) * (this->operator()(i, k) - this->operator()(j, k));
			}
			Dist(i, j) = sqrt(Dist(i, j));
		}
	}

	indiciBest.setDimension(rows, 3);

	for (int i = 0; i < rows; i++)
	{
		float min[3] = { 1.e8, 1.e8, 1.e8 };

		float jMin[3] = { -1, -1, -1 };

		for (int j = 0; j < rows; j++)
		{
			float val = Dist(i, j);
			if (val < min[0])
			{
				min[2] = min[1];
				min[1] = min[0];
				min[0] = val;

				jMin[2] = jMin[1];
				jMin[1] = jMin[0];
				jMin[0] = j;
			}
			else if (val < min[1])
			{
				min[2] = min[1];
				min[1] = val;

				jMin[2] = jMin[1];
				jMin[1] = j;
			}
			else if (val < min[2])
			{
				min[2] = val;
				jMin[2] = j;
			}
		}
		indiciBest(i, 0) = jMin[0];
		indiciBest(i, 1) = jMin[1];
		indiciBest(i, 2) = jMin[2];
	}

	return true;
}



template <class T>
void CudaMatrix<T>::print(const string& nomeFile)
{
	ofstream myfile;
	if (nomeFile != "")
	{
		myfile.open(nomeFile);
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (myfile.is_open())
			{
				myfile << data[i * cols + j] << " ";
			}
			else
			{
				cout << data[i * cols + j] << " ";
			}

		}
		if (myfile.is_open())
		{
			myfile << endl;
		}
		else
		{
			cout << endl;
		}
	}

	if (myfile.is_open())
	{
		myfile.close();
	}
}


template <class T>
void CudaMatrix<T>::printDeviceInfo();


template <class T>
void CudaMatrix<T>::printDeviceInfo()
{

	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::cout << "Device Number: " << i << std::endl;
		std::cout << "\tDevice name: " << std::string(prop.name) << std::endl;
		std::cout << "\tMemory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
		std::cout << "\tMemory Bus Width (bits): " << prop.memoryBusWidth << endl;
		std::cout << "\tPeak Memory Badwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6 << std::endl << std::endl;
		std::cout << "\tmax threads per block: " << prop.maxThreadsPerBlock << std::endl;
		std::cout << "\tmax grid size: [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
	}
}



