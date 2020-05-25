// testCudaMatrix.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//

#include <iostream>
#include <chrono>
#include "CudaMatrix.h"

using namespace std;


void provaDist()
{
	cout << "******* prova Dist *******" << endl;
	// creo la matrice dei descrittori
	int M; // 6500; // 6500;
	int N; // 128;// 128;

	cout << "M: ";
	cin >> M;
	cout << "N: ";
	cin >> N;

	bool printCout = M < 70 && N < 70;
	string fileDistances = printCout ? "" : "dist.txt";
	string fileDistancesShared = printCout ? "" : "distShared.txt";
	string fileDistancesCPU = printCout ? "" : "distCPU.txt";
	string fileIndices = printCout ? "" : "indices.txt";
	string fileIndicesShared = printCout ? "" : "indicesShared.txt";
	string fileIndicesCPU = printCout ? "" : "indicesCPU.txt";
	string fileDiff = printCout ? "" : "diff.txt";
	string fileDiffShared = printCout ? "" : "diffShared.txt";
	string fileDiffIndiciShared = printCout ? "" : "indiciDiffShared.txt";

	// ********** creazione matrice test ********************

	float* dBuffer = new float[M * N];

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			dBuffer[i * N + j] = rand() % 5;
		}
	}

	// creo il wrapper
	CudaMatrix<float> descriptorCuda(M, N, dBuffer);

	//// stampo i descrittori
	//descriptorCuda.print("descriptors.txt");

	CudaMatrix<float> distances, distancesShared, distancesCPU;
	CudaMatrix<unsigned int> indiciBest, indiciBestShared, indiciBestCPU;

	// ******************************************************

	cout << "*********************" << endl
		<< "prova gpu" << endl
		<< "*********************" << endl;

	auto start = std::chrono::steady_clock::now();
	if (!descriptorCuda.computeSelfDistancesOld(distances, indiciBest))
	{
		cout << "errore!!" << endl;
	}
	auto end = std::chrono::steady_clock::now();
	cout << "tempo impieato: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

	//distances.print(fileDiff);
	//indiciBest.print(fileIndices);


	cout << "*********************" << endl
		<< "prova gpu shared" << endl
		<< "*********************" << endl;

	start = std::chrono::steady_clock::now();
	if (!descriptorCuda.computeSelfDistances(distancesShared, indiciBestShared))
	{
		cout << "errore!!" << endl;
	}
	end = std::chrono::steady_clock::now();
	cout << "tempo impieato: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

	//distancesShared.print(fileDiffShared);
	//indiciBestShared.print(fileIndicesShared);


	// calcolo le distanze con la CPU
	cout << "*********************" << endl
		<< "prova CPU" << endl
		<< "*********************" << endl;

	start = std::chrono::steady_clock::now();
	if (!descriptorCuda.computeSelfDistancesCPU(distancesCPU, indiciBestCPU))
	{
		cout << "errore!!" << endl;
	}
	end = std::chrono::steady_clock::now();
	cout << "tempo impieato: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

	//distancesCPU.print(fileDistancesCPU);
	//indiciBestCPU.print(fileIndicesCPU);

	// *********** valuto la correttezza **************

	cout << "----------------------" << endl;

	double max = 0;
	for (int i = 0; i < distancesCPU.Rows(); i++)
	{
		for (int j = 0; j < distancesCPU.Cols(); j++)
		{
			if (abs(distancesCPU(i, j) - distances(i, j)) > max)
			{
				max = abs(distancesCPU(i, j) - distances(i, j));
			}
		}
	}

	cout << "err abs CPU GPU. abs Max: " << max << endl;

	max = 0;
	for (int i = 0; i < distancesCPU.Rows(); i++)
	{
		for (int j = 0; j < distancesCPU.Cols(); j++)
		{
			if (abs(distancesCPU(i, j) - distancesShared(i, j)) > max)
			{
				max = abs(distancesCPU(i, j) - distancesShared(i, j));
			}
		}
	}

	cout << "err abs CPU GPU shared. abs Max: " << max << endl;

	CudaMatrix<float>errGpu, errGpuShared;

	distancesCPU.difCPU(distances, errGpu);

	cout << "-------------------------------------------" << endl;
	cout << "err gpu: " << endl;
	errGpu.print(fileDiff);

	distancesCPU.difCPU(distancesShared, errGpuShared);

	cout << "-------------------------------------------" << endl;
	cout << "err gpu shared: " << endl;
	errGpuShared.print(fileDiffShared);

	CudaMatrix<unsigned int>errIndiciShared;
	indiciBestCPU.difCPU(indiciBestShared, errIndiciShared);
	cout << "---------------------------------" << endl;
	cout << "err indici shared: " << endl;
	errIndiciShared.print(fileDiffIndiciShared);

	delete[] dBuffer;

}


void provaMatMult()
{

	//CudaMatrix<int>::printDeviceInfo();

	int M;
	int N;
	int H;
	cout << "M: ";
	cin >> M;
	cout << "N: ";
	cin >> N;
	cout << "H: ";
	cin >> H;

	int dimMaxCout = 40;

	cout << "M: " << M << ", N: " << N << ", H: " << H << endl;
	bool condizioneCout = M > dimMaxCout || H > dimMaxCout;

	string fileC = condizioneCout ? "C.txt" : "";
	string fileCErrore = condizioneCout ? "CErrore.txt" : "";
	string fileCcpu = condizioneCout ? "Ccpu.txt" : "";
	string fileDiffGpu = condizioneCout ? "diffGpu.txt" : "";



	// ************** creazione matrici test ***********************
	double* aBuf = new double[M * N];
	double* bBuf = new double[N * H];

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			aBuf[i * N + j] = 135.7 * ((double)rand() / RAND_MAX); // rand() % 5; // //  (j == N - 1 && i == M - 1) ? 1 : 0; // ;i * N + j; 
		}
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < H; j++)
		{
			bBuf[i * H + j] = -45.689 * ((float)rand() / RAND_MAX);// j == i ? 1 : 0; //rand() % 3; //  //  /
		}
	}

	CudaMatrix<double> A(M, N, aBuf);
	CudaMatrix<double> B(N, H, bBuf);
	CudaMatrix<double> C, Ccpu;

	//cout << "A: " << endl;
	//A.print();
	//cout << "B: " << endl;
	//B.print(/*"B.txt"*/);

	// **********************************************************

	cout << "*********************" << endl;
	cout << "prova GPU" << endl;
	auto start = std::chrono::steady_clock::now();
	if (!A.product(B, C))
	{
		throw exception("fallito");
	}
	auto end = std::chrono::steady_clock::now();
	cout << "tempo impieato: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	cout << "C shared no multip: " << endl;
	C.print(fileC);


	cout << "*********************" << endl;
	cout << "prova CPU" << endl;
	start = std::chrono::steady_clock::now();
	if (!A.productCPU(B, Ccpu))
	{
		throw exception("fallito");
	}
	end = std::chrono::steady_clock::now();
	cout << "tempo impieato: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	cout << "C cpu: " << endl;
	Ccpu.print(fileCcpu);

	// ***************** valuto la correttezza dei risultati ******************
	cout << "-------------------------------------------" << endl;

	double max = 0;
	for (int i = 0; i < Ccpu.Rows(); i++)
	{
		for (int j = 0; j < Ccpu.Cols(); j++)
		{
			if (abs(Ccpu(i,j) - C(i,j)) > max)
			{
				max = abs(Ccpu(i, j) - C(i, j));
			}
		}
	}

	cout << "err abs CPU GPU. abs Max: " << max << endl;

	CudaMatrix<double>errGpu;

	Ccpu.difCPU(C, errGpu);

	cout << "-------------------------------------------" << endl;
	cout << "err gpu: " << endl;
	errGpu.print(fileDiffGpu);
	
	delete []aBuf;
	delete []bBuf;

}


void provaCicloSelfDist()
{
	// per M = 37000 si pianta: fallisce l'allocazione sul device e dà errore "out of memory".
	int N = 128;
	for (int M = 5000; M < 60000; M += 1000)
	{
		cout << "*******************************" << endl;
		cout << "M = " << M << endl;

		float* data = new float[M * N];

		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				data[i * N + j] = 135.7 * ((double)rand() / RAND_MAX); 
			}
		}

		
		CudaMatrix<float> descr(M, N, data);
		CudaMatrix<float> Dist;
		CudaMatrix<unsigned int> indici;

		auto start = std::chrono::steady_clock::now();
		if (!descr.computeSelfDistances(Dist, indici))
		{
			cout << "FALLITO!" << endl;
			return;
		}
		auto end = std::chrono::steady_clock::now();
		cout << "tempo impieato: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

		delete[] data;

	}
}

int main(int argc, char** argv)
{

	//CudaMatrix<int>::printDeviceInfo();

	//provaDist();
	//provaMatMult();

	provaCicloSelfDist();

	return 0;
}