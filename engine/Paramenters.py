import argparse
import sys


def parse_args():
    desc = "My project"
    parser = argparse.ArgumentParser(description=desc)#创建解析对象
    parser.add_argument('--data_file', type=str, default='Data', help='数据集的位置')
    parser.add_argument('--dataset', type=str, default='A2D2', help='数据集的位置')
    parser.add_argument('--data_process', type=bool, default=False, help='是否处理数据集')
    parser.add_argument('--MT_process', type=bool, default=1, help='是否处理数据集')
    parser.add_argument('--RIEF', type=bool, default=1, help='插值图像2X')
    parser.add_argument('--compare_sota', type=bool, default=1)
    parser.add_argument('--MRs_num', type=int, default=3)
    parser.add_argument('--MT_image_num', type=int, default=350,help="生成测试图片的数量，应该是5的倍数用于测试")
    parser.add_argument('--seg_process', type=bool, default=1, help='是否seg图像')
    parser.add_argument('--seg_type', type=str, default="OneFormer", help='三种分割图像的方法')
    parser.add_argument('--device', type=str, default="cuda", help='设备')
    parser.add_argument('--Use_time_series', type=int, default=1, help='时间序列ADS')
    parser.add_argument('--Use_vehicle_states', type=int, default=0, help='使用多模态')
    parser.add_argument('--pre_series', type=int, default=25, help='预测的时步步长5-25，可以超过25帧')#应该是5的数数，以适应序列预测的需求
    parser.add_argument('--new_test_data', type=int, default=0, help='是否使用新的测试数据')
    parser.add_argument('--pre_model', type=str, default="speed", help='测测模式 steeringorspeed')
    #parser.add_argument('--image_size', type=int, default=512, help='是使用NN还是直接使用传感器数据')
    args = parser.parse_args(sys.argv[1:])
    return args

