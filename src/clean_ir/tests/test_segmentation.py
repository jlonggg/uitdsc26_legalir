import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from segmentation import segment
from underthesea import word_tokenize


class Segmentation(unittest.TestCase):
    def test_ordinary_text_matches_pinned_underthesea(self):
        for text in ('Quy định về thuế thu nhập cá nhân năm 2026.',
                     'Nghị định 12/2024/NĐ-CP ngày 05/02/2024.',
                     'Đăng ký tại https://example.vn/ho-so và nộp phí.'):
            self.assertEqual(segment(text),word_tokenize(text,format='text'))

    def test_url_followed_by_dotted_form_field(self):
        text='Mã hồ sơ trên trang http://tuyensinh.vied.vn'+'.'*46+'\nTrình độ học vấn'
        result=segment(text)
        self.assertIn('http://tuyensinh.vied.vn',result)
        self.assertIn('học_vấn',result)
        self.assertEqual(result,segment(text))
        self.assertEqual(segment(''),'')

if __name__=='__main__':unittest.main()
