import unittest
from ncu_traffic import parse_dram_csv

class Parsing(unittest.TestCase):
    def test_wide(self):
        data='''==PROF== captured
"ID","dram__bytes_op_read.sum","dram__bytes_op_write.sum"
"","byte","byte"
"0","1,024","16"
"1","32","0"
'''
        self.assertEqual(parse_dram_csv(data), dict(kernels=2,dram_read_bytes=1056,dram_write_bytes=16))
    def test_long(self):
        data='''"ID","Metric Name","Metric Unit","Metric Value"
"0","dram__bytes_op_read.sum","byte","32"
"0","dram__bytes_op_write.sum","byte","16"
'''
        self.assertEqual(parse_dram_csv(data),dict(kernels=1,dram_read_bytes=32,dram_write_bytes=16))
        with self.assertRaises(AssertionError):
            parse_dram_csv(data.splitlines()[0]+'\n'+data.splitlines()[1])
        with self.assertRaises(AssertionError):
            parse_dram_csv(data+data.splitlines()[1]+'\n')
    def test_error(self):
        with self.assertRaises(ValueError):parse_dram_csv('==ERROR== inaccessible counters')

if __name__ == '__main__':unittest.main()
