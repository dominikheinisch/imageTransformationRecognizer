import adjacency
import heuristic
import ransac
from utils import loader
from utils import printer
from utils.timing import timing
from timeit import default_timer as timer
import numpy as np

IMAGES = {1: ['data/1/DSC03230.png', 'data/1/DSC03240.png'],
          2: ['data/2/DSC_5824.png', 'data/2/DSC_5825.png'],
          3: ['data/3/3-1.png', 'data/3/3-2.png'],
          4: ['data/4-same/4-1.png', 'data/4-same/4-2.png'],
          5: ['data/5/5-1.png', 'data/5/5-2.png'],
          6: ['data/6-diff/6-1.png', 'data/6-diff/6-2.png'],
          8: ['data/p1/p1-1.png', 'data/p1/p1-2.png'],
          9: ['data/p2/p2-1.png', 'data/p2/p2-2.png'],
          10: ['data/p3/p3-1.png', 'data/p3/p3-2.png'],
          11: ['data/p4/p4-1.png', 'data/p4/p4-2.png'],
          12: ['data/p5/p5-1.png', 'data/p5/p5-2.png'],
          13: ['data/r3/3-1.png', 'data/r3/3-2.png']
          }

PATHS = IMAGES[3]


@timing
def main():
    key_points_1, key_points_2 = loader.load_sifts(PATHS)
    pairs = adjacency.find_pairs_euclidean(key_points_1, key_points_2)

    n = 25
    t = 0.12
    filtered_pairs = adjacency.filter_pairs(pairs, n=n, threshold=t)
    ransac_ = ransac.Ransac(np.array(pairs), filtered_pairs)

    h = None
    # h = heuristic.EuclideanDistanceHeuristic(PATHS, lower_limit=0.00, upper_limit=0.3)
    # h = heuristic.ShapeHeuristic(upper_limit=0.3)
    i = 100
    size = 4
    e = 10
    start = timer()
    ransac_.calculate(size=size, no_draws=i, max_error=e, heuristic=h)
    end = timer()
    print('ransac took: {} s'.format((end - start)))
    ransac_pairs = ransac_.get_ransac_pairs()
    print('ransac pairs: {}'.format(len(ransac_pairs)))

    s = size
    # printer.print_image(PATHS, ransac_pairs, 'ransac_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e))
    printer.print_all_image(PATHS, key_points_1, key_points_2, filtered_pairs, ransac_pairs,
                            'all_ransac_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e))


if __name__ == "__main__":
    main()

# from timeit import default_timer as timer
#
# import numpy as np
#
# import adjacency
# import heuristic
# import loader
# import printer
# import ransac
# from timing import timing
#
# IMAGES = {
#     '1': ['data/1/DSC03230.png', 'data/1/DSC03240.png'],
#     '2': ['data/2/DSC_5824.png', 'data/2/DSC_5825.png'],
#     '3': ['data/3/3-1.png', 'data/3/3-2.png'],
#     '4': ['data/4/4-1.png', 'data/4/4-2.png'],
#     '5': ['data/5/5-1.png', 'data/5/5-2.png'],
#     '7': ['data/7/7-1.png', 'data/7/7-2.png'],
#     '8': ['data/8/8-1.png', 'data/8/8-2.png'],
#     'p1': ['data/p1/p1-1.png', 'data/p1/p1-2.png'],
#     'p2': ['data/p2/p2-1.png', 'data/p2/p2-2.png'],
#     'p3': ['data/p3/p3-1.png', 'data/p3/p3-2.png'],
#     'p4': ['data/p4/p4-1.png', 'data/p4/p4-2.png'],
#     'p5': ['data/p5/p5-1.png', 'data/p5/p5-2.png']
# }
#
# PATHS = IMAGES['2']
#
#
# @timing
# def main1():
#     key_points_1, key_points_2 = loader.load_sifts(PATHS)
#     start = timer()
#     pairs = adjacency.corresponding_points(key_points_1, key_points_2)
#     end = timer()
#     print('corresponding_points took: {} s'.format((end - start)))
#
#     n = 10
#     t = 0.5
#     filtered_pairs = adjacency.filter_pairs(pairs, n=n, threshold=t)
#     printer.print_image(PATHS, filtered_pairs, 'adjacency_n{}_t{}.png'.format(n, t))
#     printer.print_image(PATHS, filtered_pairs, 'adjacency_diag_n{}_t{}.png'.format(n, t), True)
#
#     if len(filtered_pairs) > 0:
#         ransac_engine = ransac.Ransac(np.array(pairs), filtered_pairs)
#
#         h = None
#         h = heuristic.EuclideanDistanceHeuristic(PATHS, lower_limit=0.00, upper_limit=0.3)
#         i = 100
#         size = 3
#         e = 10
#         start = timer()
#         ransac_engine.calculate(size=size, no_draws=i, max_error=e, heuristic=h)
#         end = timer()
#         print('ransac took: {} s'.format((end - start)))
#         ransac_pairs = ransac_engine.get_ransac_pairs()
#         print('ransac pairs: {}'.format(len(ransac_pairs)))
#
#         s = size
#         printer.print_image(PATHS, ransac_pairs, 'ransac_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e))
#         printer.print_image(PATHS, ransac_pairs, 'ransac_diag_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e), True)
#         result_path = printer.print_all_image(PATHS, key_points_1, key_points_2, filtered_pairs, ransac_pairs,
#                                               'all_ransac_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e))
#         result_path = printer.print_all_image(PATHS, key_points_1, key_points_2, filtered_pairs, ransac_pairs,
#                                               'all_ransac_diag_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e), True)
#         # webbrowser.open_new_tab(result_path)
#
#
# @timing
# def main2():
#     key_points_1, key_points_2 = loader.load_sifts(PATHS)
#     pairs = adjacency.corresponding_points(key_points_1, key_points_2)
#
#     for n in [10, 20, 30, 40, 60, 100]:
#         for t in [0.5, 0.6, 0.7, 0.75]:
#             filtered_pairs = adjacency.filter_pairs(pairs, n=n, threshold=t)
#             print(n, t, len(filtered_pairs))
#             printer.print_image(PATHS, filtered_pairs, 'adjacency_n{}_t{:.2f}.png'.format(n, t))
#             printer.print_image(PATHS, filtered_pairs, 'adjacency_diag_n{}_t{:.2f}.png'.format(n, t), True)
#
#
# @timing
# def main3():
#     key_points_1, key_points_2 = loader.load_sifts(PATHS)
#     pairs = adjacency.corresponding_points(key_points_1, key_points_2)
#
#     n = 10
#     t = 0.5
#     filtered_pairs = adjacency.filter_pairs(pairs, n=n, threshold=t)
#     printer.print_image(PATHS, filtered_pairs, 'adjacency_n{}_t{}.png'.format(n, t))
#     printer.print_image(PATHS, filtered_pairs, 'adjacency_diag_n{}_t{}.png'.format(n, t), True)
#
#     if len(filtered_pairs) > 0:
#         ransac_engine = ransac.Ransac(np.array(pairs), filtered_pairs)
#
#         h = None
#         size = 4
#
#         for i in [30, 100, 300, 1000, 3000]:
#             for e in [1, 3, 10, 30]:
#                 ransac_engine.calculate(size=size, no_draws=i, max_error=e, heuristic=h)
#                 ransac_pairs = ransac_engine.get_ransac_pairs()
#                 print('{}\t{}\t{}'.format(i, e, len(ransac_pairs)))
#
#                 s = size
#                 printer.print_image(PATHS, ransac_pairs, 'ransac_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e))
#                 printer.print_image(PATHS, ransac_pairs, 'ransac_diag_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e),
#                                     True)
#                 result_path = printer.print_all_image(PATHS, key_points_1, key_points_2, filtered_pairs, ransac_pairs,
#                                                       'all_ransac_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e))
#                 result_path = printer.print_all_image(PATHS, key_points_1, key_points_2, filtered_pairs, ransac_pairs,
#                                                       'all_ransac_diag_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e),
#                                                       True)
#                 # webbrowser.open_new_tab(result_path)
#
#
# @timing
# def main4():
#     key_points_1, key_points_2 = loader.load_sifts(PATHS)
#     pairs = adjacency.corresponding_points(key_points_1, key_points_2)
#
#     n = 10
#     t = 0.5
#     filtered_pairs = adjacency.filter_pairs(pairs, n=n, threshold=t)
#     printer.print_image(PATHS, filtered_pairs, 'adjacency_n{}_t{}.png'.format(n, t))
#     printer.print_image(PATHS, filtered_pairs, 'adjacency_diag_n{}_t{}.png'.format(n, t), True)
#
#     if len(filtered_pairs) > 0:
#         ransac_engine = ransac.Ransac(np.array(pairs), filtered_pairs)
#
#         h = None
#         size = 4
#
#         for i in [30, 100, 300, 1000, 3000]:
#             for e in [1, 3, 10, 30]:
#                 ransac_engine.calculate(size=size, no_draws=i, max_error=e, heuristic=h)
#                 ransac_pairs = ransac_engine.get_ransac_pairs()
#                 print('{}\t{}\t{}'.format(i, e, len(ransac_pairs)))
#
#                 s = size
#                 printer.print_image(PATHS, ransac_pairs, 'ransac_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e))
#                 printer.print_image(PATHS, ransac_pairs, 'ransac_diag_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e),
#                                     True)
#                 result_path = printer.print_all_image(PATHS, key_points_1, key_points_2, filtered_pairs, ransac_pairs,
#                                                       'all_ransac_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e))
#                 result_path = printer.print_all_image(PATHS, key_points_1, key_points_2, filtered_pairs, ransac_pairs,
#                                                       'all_ransac_diag_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e),
#                                                       True)
#                 # webbrowser.open_new_tab(result_path)
#
#
# if __name__ == "__main__":
#     main4()
#     # for name, value in IMAGES.items():
#     #     print(name)
#     #     PATHS = value
#     #     main()