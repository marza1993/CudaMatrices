#pragma once

#ifdef CUDAMATRIX_EXPORTS
#define CUDAMATRIX_API __declspec(dllexport)
#else
#define CUDAMATRIX_API __declspec(dllimport)
#endif

#include <string>
#include <iostream>
#include <sstream>


template<class T>
class CudaMatrixSimmetric;


// classe wrapper di una matrice che permette di effettuare alcune operazioni in modo parallelo
template <class T>
class CUDAMATRIX_API CudaMatrix
{

protected:

	size_t rows;
	size_t cols;

	T* data = nullptr;

	bool ownsData = true;

public:

	CudaMatrix()
	{

#ifdef _DEBUG
		std::cout << "default constructor" << std::endl;
#endif
		this->rows = 0;
		this->cols = 0;
	}

	CudaMatrix(size_t rows, size_t cols) : rows(rows), cols(cols)
	{

#ifdef _DEBUG
		std::cout << "constructor con NEW" << std::endl;
#endif
		data = new T[rows * cols];
	};


	// costruttore in cui il buffer con i dati sono passati dall'esterno, e quindi l'oggetto corrente
	// non ha il possesso dei dati (perciò non effettuerà il delete nel distruttore). "Aggregation"
	CudaMatrix(size_t rows, size_t cols, T* data) : rows(rows), cols(cols), data(data)
	{

#ifdef _DEBUG
		std::cout << "constructor con dati esterni, ownsdata = false" << std::endl;
#endif
		ownsData = false;
	};


	// virtual destructor per evenutale ereditarietà
	virtual ~CudaMatrix()
	{

#ifdef _DEBUG
		std::cout << "destructor" << std::endl;
#endif
		// il controllo del nullptr è superfluo (non farebbe nulla il delete[])
		if (ownsData && data != nullptr)
		{

#ifdef _DEBUG
			std::cout << "DELETE dei dati" << std::endl;
#endif
			delete[] data;
		}
	}

	virtual void clearData()
	{

#ifdef _DEBUG
		std::cout << "clearData" << std::endl;
#endif
		if (ownsData && data != nullptr)
		{

#ifdef _DEBUG
			std::cout << "DELETE dei dati" << std::endl;
#endif
			delete[] data;
			// imposto a null il puntatore ai dati per evitare che si possa fare un delete di nuovo
			data = nullptr;
		}
	}

	virtual T& operator()(size_t i, size_t j)
	{
		if (i >= rows || j >= cols)
		{
			std::stringstream errorMsg;
			errorMsg << "Indice passato oltre la dimensione della matrice!" << std::endl <<
				"dim matrice: (" << rows << ", " << cols << ")" << std::endl
				<< "posizione richiesta: (" << i << ", " << j << ")" << std::endl;

			throw std::exception(errorMsg.str().c_str());
		}
		return data[i * cols + j];
	}

	bool sum(const CudaMatrix& other, CudaMatrix& result);

	bool dif(const CudaMatrix& other, CudaMatrix& result);

	bool difCPU(const CudaMatrix& other, CudaMatrix& result);

	bool product(const CudaMatrix& other, CudaMatrix& result);

	bool productCPU(const CudaMatrix& other, CudaMatrix& result);

	bool linearCombination(const CudaMatrix& other, const float alpha, const float beta, CudaMatrix& result);

	bool computeSelfDistances(CudaMatrixSimmetric<float>& Dist, CudaMatrix<unsigned int>& indiciBest);

	bool computeSelfDistancesCPU(CudaMatrix<float>& Dist, CudaMatrix<unsigned int>& indiciBest);

	virtual void print(const std::string& nomeFile = "");

	static void printDeviceInfo();


	T* getData() const
	{
		return data;
	}

	size_t Rows() const
	{
		return rows;
	}

	size_t Cols() const
	{
		return cols;
	}

	bool isDataOwned()
	{
		return ownsData;
	}

	virtual void setDimension(size_t rows, size_t cols)
	{
		// sarebbe meglio mettere un check su onwsdata??

		// se erano già stati allocati dei dati, li elimino
		if (data != nullptr && ownsData)
		{

#ifdef _DEBUG
			std::cout << "set Dimension, i dati erano già stati allocati, faccio DELETE, per liberare la vecchia zona di memoria, " <<
				"altrimenti avrei un memory leak" << std::endl;
#endif
			delete[] data;
		}

		this->rows = rows;
		this->cols = cols;

#ifdef _DEBUG
		std::cout << "set Dimension, faccio NEW" << std::endl;
#endif
		data = new T[this->rows * this->cols];
	}

};


// forward declaration
template class CudaMatrix<int>;
template class CudaMatrix<unsigned char>;
template class CudaMatrix<unsigned int>;
template class CudaMatrix<float>;
template class CudaMatrix<double>;



