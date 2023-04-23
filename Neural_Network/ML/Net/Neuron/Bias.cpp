#include "Net/Layer/Layer.h"

namespace ML
{
	void Bias::init(uInt numConnections)
	{
		m_connections.resize(numConnections);
	}

	DEFINE_ITERATORS(Connection<0>, Bias, m_connections);

	void Bias::calcGradient(const Layer& nextLayer)
	{
		for (const Neuron& n : nextLayer)
			(*this)[n.getIndex()].DERIVATIVE += n.getGradient();
	}

	void Bias::applyGradients(mFloat lr, mFloat alpha, uInt noDataValues)
	{
		for (auto& cn : *this)
			cn.update(lr,alpha, noDataValues);
	}

}