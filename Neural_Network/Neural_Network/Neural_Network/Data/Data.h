#pragma once

#include <cstdint>

namespace ML
{
    using uInt = std::uint32_t;;

    template <typename Dat>
    class Vec
    {
    public:
        Vec(uInt len)
            :m_data(new Dat[len]), m_len(len)
        {
        }

        virtual ~Vec()
        {
            if (m_data)
                delete[] m_data;
        }

        Vec(const Vec& copy)
            :m_len(copy.m_len), m_data(new Dat[copy.m_len])
        {
            uInt i = 0;
            for (const auto& el : copy)
                (*this)[i++] = el;
        }

        Vec(Vec&& move)
            :m_len(move.m_len), m_data(move.m_data)
        {
            move.m_data = nullptr;
            move.m_len = 0;
        }

        Dat& operator[](uInt i)
        {
            assert(i < m_len);

            return m_data[i];
        }

        const Dat& operator[](uInt i) const
        {
            assert(i < m_len);

            return m_data[i];
        }

        Dat* begin()
        {
            return m_data;
        }

        Dat* end()
        {
            return m_data + m_len;
        }        
        
        const Dat* begin() const
        {
            return m_data;
        }

        const Dat* end() const
        {
            return m_data + m_len;
        }

        uInt getLen() const
        {
            return m_len;
        }

        const Dat& back() const 
        {
            return m_data[m_len - 1];
        }        
        
        Dat& back()
        {
            return m_data[m_len - 1];
        }

    private:
        uInt m_len;

    protected:
        Dat* m_data;

    };
 
    template <typename Dat>
    class Var : Vec<Dat> 
    {
    public:
        Var(uInt num = 1)
            :Vec<Dat>(num == 0 ? 0 : 1)
        {
        }

        Var(const Var& copy)
            :Vec<Dat>(1)
        {
            *m_data = *copy.m_data;
        }

        Var(Var&& move)
            :m_data(move.m_data)
        {
            move.m_data = nullptr;
        }

        operator Dat& ()
        {
            return *m_data;
        }

        operator Dat* ()
        {
            return m_data;
        }

        operator bool()
        {
            return m_data;
        }

    };

 
}