#pragma once
#include "CudaMatrix.h"
#include <fstream>

// classe che rappresenta una matrice simmetrica (non sparsa). Ottimizza l'utilizzo della memoria
// occupando (N * (N - 1)) / 2 elementi invece che N * N (utile per ridurre l'occupazione poi sulla GPU).
template <class T>
class CUDAMATRIX_API CudaMatrixSimmetric : public CudaMatrix<T>
{

public:

	CudaMatrixSimmetric()
	{

#ifdef _DEBUG
		std::cout << "default constructor Simmetric" << std::endl;
#endif
		this->rows = 0;
	}

	CudaMatrixSimmetric(size_t rows)
	{
		this->rows = rows;
#ifdef _DEBUG
		std::cout << "constructor Simmetric con NEW" << std::endl;
#endif
		// NB: per usare i membri della classe base con i template è necessario anteporre this->
		this->data = new T[(rows * (rows + 1)) / 2];
	};


	// costruttore in cui il buffer con i dati sono passati dall'esterno, e quindi l'oggetto corrente
	// non ha il possesso dei dati (perciò non effettuerà il delete nel distruttore). "Aggregation"
	CudaMatrixSimmetric(size_t rows, T* data)
	{
		this->rows = rows;
		this->data = data;

#ifdef _DEBUG
		std::cout << "constructor con dati esterni, ownsdata = false" << std::endl;
#endif
		this->ownsData = false;
	};


	// virtual destructor per evenutale ereditarietà
	virtual ~CudaMatrixSimmetric()
	{

#ifdef _DEBUG
		std::cout << "destructor Simmetric" << std::endl;
#endif

	}


	T& operator()(size_t i, size_t j)
	{
		if (i >= this->rows || j >= this->rows)
		{
			std::stringstream errorMsg;
			errorMsg << "Indice passato oltre la dimensione della matrice!" << std::endl <<
				"dim matrice: (" << this->rows << ", " << this->rows << ")" << std::endl
				<< "posizione richiesta: (" << i << ", " << j << ")" << std::endl;

			throw std::exception(errorMsg.str().c_str());
		}
		if (i >= j)
		{
			return this->data[(i * (i + 1) / 2) + j];
		}
		else
		{
			return this->data[(j * (j + 1) / 2) + i];
		}
	}


	void setDimension(size_t rows)
	{

		// se erano già stati allocati dei dati, li elimino
		if (this->data != nullptr && this->ownsData)
		{

#ifdef _DEBUG
			std::cout << "Simmetric: set Dimension, i dati erano già stati allocati, faccio DELETE, per liberare la vecchia zona di memoria, " <<
				"altrimenti avrei un memory leak" << std::endl;
#endif
			delete[] this->data;
		}

		this->rows = rows;

#ifdef _DEBUG
		std::cout << "Simmetric: set Dimension, faccio NEW" << std::endl;
#endif
		this->data = new T[(rows * (rows + 1)) / 2];
	}


	void print(const std::string& nomeFile = "")
	{
#ifdef _DEBUG
		std::cout << "print Simmetric" << std::endl;
#endif

		std::ofstream myfile;
		if (nomeFile != "")
		{
			myfile.open(nomeFile);
		}

		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->rows; j++)
			{
				if (myfile.is_open())
				{
					int indice = (i * (i + 1) / 2) + j;
					if (i < j)
					{
						indice = (j * (j + 1) / 2) + i;
					}
					myfile << this->data[indice] << " ";
				}
				else
				{
					int indice = (i * (i + 1) / 2) + j;
					if (i < j)
					{
						indice = (j * (j + 1) / 2) + i;
					}
					std::cout << this->data[indice] << " ";
				}

			}
			if (myfile.is_open())
			{
				myfile << std::endl;
			}
			else
			{
				std::cout << std::endl;
			}
		}

		if (myfile.is_open())
		{
			myfile.close();
		}
	}

};




// forward declaration
template class CudaMatrixSimmetric<int>;
template class CudaMatrixSimmetric<unsigned char>;
template class CudaMatrixSimmetric<unsigned int>;
template class CudaMatrixSimmetric<float>;
template class CudaMatrixSimmetric<double>;

