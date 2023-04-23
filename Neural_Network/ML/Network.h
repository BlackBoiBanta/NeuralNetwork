#pragma once 
#include "Net/Layer/Layer.h"
#include <sstream>

namespace ML
{

	class Network
	{
	public:
		Network(const NetConfig&);

		uInt getLen() const;

		template <typename Dat>
		void feedInFloats(const Dat* data, uInt size) 
		{

			if (sizeof(Dat) > sizeof(float))
				std::cout << "There will be a possible loss of data, as your input data type is larger than the sizeof a float.\n";

			uInt i = 0;
			for (Neuron& n : getInputLayer())
			{
				if (i == size)
					assert(false && "The input data should be a vector of equal length to the input layer...\n");

				n.setValue(static_cast<float>(*(data + i)));
				n.setOutput(getInputLayer().actFunc(n.getValue()));

				i++;
			}
		}

		Layer& getInputLayer();
		Layer& getOutputLayer();

		const Layer& getInputLayer() const;
		const Layer& getOutputLayer() const;

		void propagate();
		void backProp(); 
		void calculate(); 
		float getSSR() const;
		void setCorrectVal(uInt index, float val);

		DECLARE_ITERATORS(Layer, Network);

		friend const Vec<float>& getOutputVec(const Network&, bool changedInputs);

	private:
		Vec<Layer> m_layers;

		float m_SSR;
		float m_trueVal;
		uInt m_numberOfDataValues;
		NetConfig m_config;

		float getCorrectVal(uInt index) const;

	};

	inline const Vec<float>& getOutputVec(const Network& net, bool changedInputs)
	{
		static Vec<float> outputVec(net.m_layers.back().getLen());

		if (!changedInputs) // if they've given the nn some new inputs 
			return outputVec;

		uInt i = 0;
		for (const auto& n : net.m_layers.back())
			outputVec[i++] = n.getOutput();

		return outputVec;

	}


}

inline std::ostream& operator<<(std::ostream&, const ML::Network& net)
{
	const auto& outputVec = ML::getOutputVec(net, true);
	for (const auto& output : outputVec)
		std::cout << "{ " << output << "}, " << "\n";

	return std::cout;
}




























// i need a function that set's all the static variables back to normal, so that I can initialize more network objects
