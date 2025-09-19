#include <string>
#include <cstring>
#include <vector>
#include <array>
#include <cassert>
#include <iostream>
#include <fstream>
#include <bit>
#include <cstdint>

// Comment out if you want the filename hardcoded.
#define OPEN_FILE_ARG

using namespace std;

const int32_t CARDS_IN_SET = 3;
const int32_t ATTRIBUTE_OPTION_COUNT = 3;
const int32_t BITS_PER_ATTR = ATTRIBUTE_OPTION_COUNT;
const int32_t MAX_ATTRIBUTE_STR_LEN = 10;

const int32_t BITMASK = (1 << BITS_PER_ATTR) - 1;
const int32_t SINGLE_SHIFT = 8;
const int32_t Z_SHIFT = 2 * SINGLE_SHIFT;
const int32_t Y_SHIFT = 1 * SINGLE_SHIFT;

const int32_t CUDA_BLOCK_COUNT = 2;

const uint32_t CARD_MASK = 0xFF;

enum ATTRIBUTE_NAMES
{
    SHAPE,
    COLOR,
    FILLING,
    COUNT,
    ATTRIBUTE_COUNT
};
const string ATTRIBUTE_NAMES_STR[ATTRIBUTE_NAMES::ATTRIBUTE_COUNT] = {
    "shape",
    "color",
    "filling",
    "count"};

// If you change a name you need to also change it in the
// const arrays of find_option_CUDA(char *attr) and attr_to_uint(int32_t curr) AND SORT AGAIN!
// Also notice MAX_ATTRIBUTE_STR_LEN above.
const string ATTRIBUTES[ATTRIBUTE_NAMES::ATTRIBUTE_COUNT][ATTRIBUTE_OPTION_COUNT] = {
    {"rhombus", "squiggle", "oval"},
    {"red", "green", "blue"},
    {"empty", "striped", "full"},
    {"1", "2", "3"}};

// A card is represented by 12 bits, 3 for every attribute.
// Every attribute can have the value of 0b001, 0b010 or 0b100 which corresponds to it's options.
__device__ int32_t finish_flag = 0;
__global__ void is_set_kernel(uint32_t *table, uint32_t table_size, uint32_t *ans)
{
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;         // First card index
    int32_t y = blockIdx.y * blockDim.y + threadIdx.y + x + 1; // Second card index
    int32_t z = blockIdx.z * blockDim.z + threadIdx.z + y + 1; // Third card index

    if (z >= table_size)
    {
        // If we chose cards that don't exist
        return;
    }

    // We use bitwise calculations that gives us the following possibilities for each 3 bits:
    // 0b000 => All the same option OR All different options
    // Anything else => There are 2 categories and we rule out this combination.
    uint32_t a = table[x];
    uint32_t b = table[y];
    uint32_t c = table[z];
    // This calculation was suggested by ChatGPT.
    uint32_t all_calc = ((a & b) | (a & c) | (b & c)) & ~(a & b & c);

    if (all_calc == 0)
    {
        int32_t old = atomicCAS(&finish_flag, 0, 1);
        if (old == 0)
        {
            // First thread
            *ans = ((x) | (y << Y_SHIFT) | (z << Z_SHIFT));
        }
    }
}

__device__ char tolower(char c)
{
    constexpr int8_t ADD = 'a' - 'A';
    if (c <= 'Z' && c >= 'A')
        return c + ADD;
    return c;
}

__device__ int32_t string_compare(const char *a, const char *b)
{
    int32_t i;
    char ai, bi;
    for (i = 0; a[i] != '\0' && b[i] != '\0'; i++)
    {
        ai = tolower(a[i]);
        bi = tolower(b[i]);

        if (ai < bi)
        {
            return 1;
        }
        if (ai > bi)
        {
            return -1;
        }
    }
    if (a[i] == '\0' && b[i] != '\0')
    {
        return 1;
    }
    if (a[i] != '\0' && b[i] == '\0')
    {
        return -1;
    }
    return 0;
}

__device__ uint32_t attr_to_uint(int32_t curr)
{
    // Sorted parallel arrays for binary search.
    // Both parallel to ATTRIBUTE_OPTS_CUDA in find_option_CUDA(char *).
    // Attribute index
    const int32_t ATTRIBUTE_INDEX_CUDA[] = {ATTRIBUTE_NAMES::COUNT, ATTRIBUTE_NAMES::COUNT, ATTRIBUTE_NAMES::COUNT,
                                            ATTRIBUTE_NAMES::COLOR, ATTRIBUTE_NAMES::FILLING, ATTRIBUTE_NAMES::FILLING,
                                            ATTRIBUTE_NAMES::COLOR, ATTRIBUTE_NAMES::SHAPE, ATTRIBUTE_NAMES::COLOR,
                                            ATTRIBUTE_NAMES::SHAPE, ATTRIBUTE_NAMES::SHAPE, ATTRIBUTE_NAMES::FILLING};
    // Option index in attribute
    const int32_t ATTRIBUTE_OPT_INDEX_CUDA[] = {0, 1, 2, 2, 0, 2, 1, 2, 0, 0, 1, 1};
    return (ATTRIBUTE_INDEX_CUDA[curr] | (ATTRIBUTE_OPT_INDEX_CUDA[curr] << SINGLE_SHIFT));
}

__device__ uint32_t find_option_CUDA(char *attr)
{
    // Sorted parallel arrays for binary search.
    // This is parallel to ATTRIBUTE_INDEX_CUDA and ATTRIBUTE_OPT_INDEX_CUDA in attr_to_uint(int32_t).
    // Option name array
    const char *ATTRIBUTE_OPTS_CUDA[] = {"1", "2", "3", "blue", "empty", "full", "green", "oval", "red", "rhombus", "squiggle", "striped"};

    // Binary search
    bool found = false;
    int32_t min = 0, max = ATTRIBUTE_COUNT * ATTRIBUTE_OPTION_COUNT - 1;
    int32_t curr;
    while (min < max)
    {
        curr = min + (max - min) / 2;
        int32_t rel = string_compare(ATTRIBUTE_OPTS_CUDA[curr], attr);

        if (rel > 0)
        {
            min = curr + 1;
        }
        else if (rel < 0)
        {
            max = curr - 1;
        }
        else
        {
            found = true;
            break;
        }
    }

    if (found)
    {
        return attr_to_uint(curr);
    }
    if (string_compare(ATTRIBUTE_OPTS_CUDA[min], attr) == 0)
        return attr_to_uint(min);
    if (string_compare(ATTRIBUTE_OPTS_CUDA[max], attr) == 0)
        return attr_to_uint(max);
    return ~0;
}

// cuda_cards_str is an array of strings, each string has ATTRIBUTE_COUNT parts seaparated by a space
// indices is an array of the indices of the string parts in the strings
// table_size Cards (typically 12) * 4 Attributes each = 48 String parts
// ans should have a place for table_size cards.
__global__ void encode_card_kernel(char **cuda_cards_str, int32_t *indices, uint32_t *ans, int32_t *encode_fail, int32_t *encode_fail_general)
{
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x; // Card index
    int32_t y = blockIdx.y * blockDim.y + threadIdx.y; // Attribute index

    // Index in indices array
    int32_t index = x * ATTRIBUTE_COUNT + y;
    // If this is the first attribute in the string, we take the string from 0,
    // otherwise we take the string from the previous index.
    int32_t from;
    if (y > 0)
        from = indices[index - 1] + 1;
    else
        from = 0;
    int32_t to = indices[index];
    uint32_t opt;

    if (from > to)
    {
        // ERROR
        opt = ~0;
    }
    else
    {
        char *attr = new char[MAX_ATTRIBUTE_STR_LEN];
        // Copy string and end with a '\0'
        memcpy(attr, cuda_cards_str[x] + from, to - from);
        attr[to - from] = '\0';

        opt = find_option_CUDA(attr);
        delete[] attr;
    }

    // Check if succeeded
    if (opt == ~0)
    {
        atomicOr(encode_fail + x, 1 << y);
        atomicExch(encode_fail_general, true);
        return;
    }

    atomicOr(ans + x, 1 << (ATTRIBUTE_OPTION_COUNT * (opt & CARD_MASK) + (opt >> SINGLE_SHIFT)));
}

// Find a character in an array of strings and write to an array of indices (splitted).
// max_in_str is the maximum instances of the character to look for.
__global__ void split_str_kernel(char **str, char ch, int32_t *splitted, int32_t max_in_str)
{
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x; // String index

    int32_t count = 0;
    int32_t i;
    // int32_t substr_index = 0;
    for (i = 0; str[x][i] != '\0' && count < max_in_str; i++)
    {
        if (str[x][i] == ch)
        {
            splitted[max_in_str * x + (count++)] = i;
        }
    }
    if (count >= max_in_str)
        splitted[max_in_str * x + count] = -1;
    else
        splitted[max_in_str * x + count] = i;
}

// Decode on CPU
__host__ string decode_card(uint32_t card)
{
    string ans;
    for (int32_t i = 0; i < ATTRIBUTE_COUNT; i++)
    {
        string attr = ATTRIBUTE_NAMES_STR[i];
        int32_t opt_idx = (card >> (ATTRIBUTE_OPTION_COUNT * i)) & BITMASK;
        string opt = ATTRIBUTES[i][__countr_zero(opt_idx)];

        ans += attr + ": " + opt + ", ";
    }

    return ans;
}

int32_t main(int32_t argc, char **argv)
{
#ifdef OPEN_FILE_ARG
    if (argc < 2)
    {
        cerr << "Please include filename." << endl;
        return -1;
    }
    ifstream read_file(argv[1]);
#else
    ifstream read_file("data.txt");
#endif

    // Open file and save its content to a vector
    vector<char *> cards_str;

    string curr_str;
    while (getline(read_file, curr_str))
    {
        char *tmp_card_ptr;
        cudaMalloc(&tmp_card_ptr, curr_str.length() + 1);
        cudaMemcpy(tmp_card_ptr, curr_str.c_str(), curr_str.length() + 1, cudaMemcpyKind::cudaMemcpyHostToDevice);
        cards_str.push_back(tmp_card_ptr);
    }

    read_file.close();

    int32_t table_size = cards_str.size();

    // Find spaces in strings
    char **cuda_cards_str;
    cudaMalloc(&cuda_cards_str, table_size * sizeof(char *));
    cudaMemcpy(cuda_cards_str, cards_str.data(), table_size * sizeof(char *), cudaMemcpyKind::cudaMemcpyHostToDevice);
    int32_t *splitted_indices;
    cudaMalloc(&splitted_indices, ATTRIBUTE_COUNT * table_size * sizeof(int32_t));
    split_str_kernel<<<1, table_size>>>(cuda_cards_str, ' ', splitted_indices, ATTRIBUTE_COUNT);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaError_t::cudaSuccess)
    {
        cerr << cudaGetErrorString(err) << endl;
        for (char *cs : cards_str)
        {
            cudaFree(cs);
        }
        cudaFree(cuda_cards_str);
        cudaFree(splitted_indices);
        return -1;
    }

    cudaDeviceSynchronize();

    // Encode cards
    dim3 encode_threads(table_size, ATTRIBUTE_COUNT, 1);
    dim3 encode_blocks(1, 1, 1);
    uint32_t *table;
    cudaMalloc(&table, table_size * sizeof(uint32_t));
    cudaMemset(table, 0, table_size * sizeof(uint32_t));
    int32_t *encode_fail_CUDA;
    cudaMalloc(&encode_fail_CUDA, table_size * sizeof(int32_t));
    cudaMemset(encode_fail_CUDA, 0, table_size * sizeof(int32_t));
    int32_t *encode_fail_general_CUDA;
    cudaMalloc(&encode_fail_general_CUDA, 1 * sizeof(int32_t));
    cudaMemset(encode_fail_general_CUDA, 0, 1 * sizeof(int32_t));
    encode_card_kernel<<<encode_blocks, encode_threads>>>(cuda_cards_str, splitted_indices, table, encode_fail_CUDA, encode_fail_general_CUDA);

    // Check for CUDA errors
    err = cudaGetLastError();
    if (err != cudaError_t::cudaSuccess)
    {
        cerr << cudaGetErrorString(err) << endl;
        for (char *cs : cards_str)
        {
            cudaFree(cs);
        }
        cudaFree(cuda_cards_str);
        cudaFree(splitted_indices);
        cudaFree(table);
        cudaFree(encode_fail_CUDA);
        cudaFree(encode_fail_general_CUDA);
        return -1;
    }

    cudaDeviceSynchronize();

    // Check for encoding errors
    int32_t encode_fail_general;
    cudaMemcpy(&encode_fail_general, encode_fail_general_CUDA, 1 * sizeof(int32_t), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if (encode_fail_general != 0)
    {
        // We have errors.
        // Don't continue!
        int32_t *encode_fail = new int32_t[table_size];
        cudaMemcpy(encode_fail, encode_fail_CUDA, table_size * sizeof(int32_t), cudaMemcpyKind::cudaMemcpyDeviceToHost);
        for (int32_t i = 0; i < table_size; i++)
        {
            if (encode_fail[i] != 0)
            {
                for (int32_t j = 0; j < ATTRIBUTE_COUNT; j++)
                {
                    if (((encode_fail[i] >> j) & 0b1) == 0b1)
                        cerr << "Error in row " << i + 1 << ", in attribute number " << j + 1 << "." << endl;
                }
            }
        }

        // Free all
        for (char *cs : cards_str)
        {
            cudaFree(cs);
        }
        cudaFree(cuda_cards_str);
        cudaFree(splitted_indices);
        cudaFree(table);
        cudaFree(encode_fail_CUDA);
        cudaFree(encode_fail_general_CUDA);
        delete[] encode_fail;

        return -1;
    }

    // Free
    int32_t *cards = new int32_t[table_size];
    cudaMemcpy(cards, table, table_size * sizeof(int32_t), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(splitted_indices);
    for (char *cs : cards_str)
    {
        cudaFree(cs);
    }
    cudaFree(cuda_cards_str);
    cudaFree(encode_fail_CUDA);
    cudaFree(encode_fail_general_CUDA);

    // Prepare CUDA primary algorithm
    uint32_t *ans;
    cudaMalloc(&ans, 1 * sizeof(uint32_t));
    cudaMemset(ans, 0, sizeof(uint32_t));

    // CUDA dimentions, there are more threads than needed.
    dim3 set_threads(table_size / CUDA_BLOCK_COUNT + 1, table_size / CUDA_BLOCK_COUNT + 1, table_size / CUDA_BLOCK_COUNT + 1);
    dim3 set_blocks(CUDA_BLOCK_COUNT, CUDA_BLOCK_COUNT, CUDA_BLOCK_COUNT);

    // CUDA primary algorithm
    is_set_kernel<<<set_blocks, set_threads>>>(table, table_size, ans);

    // After CUDA primary algorithm
    err = cudaGetLastError();
    if (err != cudaError_t::cudaSuccess)
    {
        cerr << cudaGetErrorString(err) << endl;
        cudaFree(splitted_indices);
        cudaFree(table);
        return -1;
    }

    cudaDeviceSynchronize();
    cudaFree(table);
    uint32_t host_ans;
    cudaMemcpy(&host_ans, ans, 1 * sizeof(uint32_t), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(ans);

    // Decode result
    if (host_ans == (uint32_t)0)
        cout << "No set found!" << endl;
    else
        for (int32_t i = 0; i < CARDS_IN_SET; i++)
        {
            uint32_t card_idx = host_ans & CARD_MASK;
            cout << card_idx + 1 << ": " << decode_card(cards[card_idx]) << endl;
            host_ans >>= SINGLE_SHIFT;
        }

    delete[] cards;

    return 0;
}