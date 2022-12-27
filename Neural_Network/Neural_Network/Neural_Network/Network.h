#pragma once 
#include "Network/Layer.h"
#include "Network/NetConfig.h"

namespace ML
{
	class Network
	{
	public:
		Network();

		uInt size() const;
		float getSSR() const;
		void setTrueVal(uInt index);
		float getCorrectVal(uInt index) const;

		NETWORK_T_VARS;
		DECLARE_ITERATORS(Layer, Network);


		template <typename Dat>
		void feedInFloats(const Dat* data, uInt size)
		{
			Layer& inputLayer = *(begin());
			Layer& outputLayer = *(end() - 1);

			if (sizeof(Dat) > sizeof(float))
				std::cout << "There will be a possible loss of data, as your input data type is larger than the sizeof a float.\n";

			uInt i = 0;
			for (Neurone& n : inputLayer)
			{
				if (i == size)
					assert(false && "The input data should be a vector of equal length to the input layer...\n");

				n.setValue(static_cast<float> (*(data + i)));

				i++;
			}

			for (Layer& l : *this)
				l.propagateOutputs();

		}

		operator bool();							// training
		void backProp();							// training
		void applyDeltaWeights();                   // training                
		void train(const std::string& correctVal);	// training

	private:
		float m_SSR;
		uInt m_trueValIndex;
		Vec<Layer> m_layers;
	};


}

inline std::ostream& operator<<(std::ostream&, const ML::Network& net)
{
	for (const auto& n : net[net.getSize() - 1])
	{
		std::cout << "{ " << n.getValue() << " }\n";
	}

	return std::cout;
}

























// i need a function that set's all the static variables back to normal, so that I can initialize more network objects
