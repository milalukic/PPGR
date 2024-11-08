import numpy as np
import imutils
import cv2
import math

class Panorama:
    def stitch_together(self, imgs, ratio=0.75):
        (img_left, img_right) = imgs
        (points_left, features_left) = self.find(img_left)
        (points_right, features_right) = self.find(img_right)

        # Matching the corresponding points on pictures, determining
        # the matrix M with RANSAC.
        M = self.match_points(points_right, points_left, features_right, features_left, ratio)

        # Wrong format / not enough points to pair them up.
        if M is None:
            return None

        (pairs, homography_matrix, points) = M

        # Drawing the matched points and find the leftmost detected point.
        (pair_img, min_point) = self.draw_matched(img_right, img_left, points_right, points_left, pairs, points)

        # Calculate panorama width
        panorama_width = self.width(img_right.shape[1], img_left.shape[1], min_point)

        # Form the panorama by warping the right image
        panorama = cv2.warpPerspective(img_right, homography_matrix, (panorama_width, img_right.shape[0]))
        panorama[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
        return (panorama, pair_img)


    # Converts the image to grayscale, then uses SIFT (Scale-Invariant Feature Transform) to detect keypoints and features.
    # Converting to grayscale simplifies the image by removing color information, focusing solely on the intensity variations, which are key for detecting stable keypoints and features across different scales and rotations.
    def find(self, img):
        # Convert to greyscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feature = cv2.xfeatures2d.SIFT_create()
        (points, features) = feature.detectAndCompute(img, None)
        points = np.float32([point.pt for point in points])
        return (points, features)

    def match_points(self, points_right, points_left, features_right, features_left, ratio):
        # Initial matches (Brute force - K Nearest Neighbours for k=2).
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        initial_matches = matcher.knnMatch(features_right, features_left, k=2)

        # Filtering by using Lowe's ratio test
        matches = [
            (match[0].trainIdx, match[0].queryIdx)
            for match in initial_matches
            if len(match) == 2 and match[0].distance < match[1].distance*ratio
        ]

        if len(matches) >= 4:
            points_right_filtered = np.float32([points_right[i] for (_, i) in matches])
            points_left_filtered = np.float32([points_left[i] for (i, _) in matches])

            # Calculating the correct points by using RANSAC and DLT algorithm
            _, points = cv2.findHomography(points_right_filtered, points_left_filtered, cv2.RANSAC, 4.0)
            M = self.DLT_normalized(points_right_filtered, points_left_filtered, points)

            return matches, M, points

        return None

    def draw_matched(self, img_right, img_left, points_right, points_left, matches, points):
        # Stitching the images together
        height, width_l, width_r = max(img_left.shape[0], img_right.shape[0]), img_left.shape[1], img_right.shape[1]

        pair_img = np.zeros((height, width_l+width_r, 3), dtype="uint8")
        pair_img[0:img_left.shape[0], 0:width_l] = img_left
        pair_img[0:img_right.shape[0], width_l:] = img_right

        min_point = (width_l + width_r, 0)

        for ((i, j), point) in zip(matches, points):
            point_l = tuple(map(int, points_left[i]))
            point_d = (int(points_right[j][0] + width_l), int(points_right[j][1]))

            # Colors based on correctness
            color = (0, 0, 255) if point == 1 else (255, 0, 0)
            cv2.circle(pair_img, point_l, radius=0, color=color, thickness=3)
            cv2.circle(pair_img, point_d, radius=0, color=color, thickness=3)

            # Leftmost point
            if point == 1 and point_l[0] < min_point[0]:
                min_point = point_l

        return pair_img, min_point

    # Helper functions
    def width(self, width_r, width_l, min_point):
        return width_r + width_l - (width_l - min_point[0])

    def normalize_matrix(self, M):
        if M[2][2] != 1 and M[2][2] != 0:
            M = M/M[2][2]
        return M

    # Calculating the 2x9 matrix based on the lemmas of DLT algorithm
    def dlt_lemma(self, M, N):
        M = np.array([M[0], M[1], 1])
        N = np.array([N[0], N[1], 1])
        return [
            [0, 0, 0, -N[2] * M[0], -N[2] * M[1], -N[2] * M[2], N[1] * M[0], N[1] * M[1], N[1] * M[2]],
            [N[2] * M[0], N[2] * M[1], N[2] * M[2], 0, 0, 0, -N[0] * M[0], -N[0] * M[1], -N[0] * M[2]]
        ]

    def DLT(self, origs, imgs):
        M = []
        for orig, img in zip(origs, imgs):
            m = self.dlt_lemma(orig, img)
            M.extend(m)
        _, _, V = np.linalg.svd(M)
        return self.normalize_matrix(V[-1].reshape(3, 3))

    def normalize(self, points):
        X, Y = np.mean(points, axis=0)
        G = [[1, 0, -X], [0, 1, -Y], [0, 0, 1]]
        r = np.mean(np.sqrt((points - [X, Y]) ** 2).sum(axis=1))
        S = [[math.sqrt(2) / r, 0, 0], [0, math.sqrt(2) / r, 0], [0, 0, 1]]
        return np.dot(S, G)

    def DLT_normalized(self, origs, imgs, points):
        T = self.normalize(origs)
        TP = self.normalize(imgs)
        originals = [np.dot(T, [o[0], o[1], 1])[:2] for o, t in zip(origs, points) if t]
        images = [np.dot(TP, [s[0], s[1], 1])[:2] for s, t in zip(imgs, points) if t]
        PP = self.DLT(originals, images)
        return self.normalize_matrix(np.dot(np.dot(np.linalg.inv(TP), PP), T))

if __name__ == "__main__":
    # Load
    left = cv2.imread('left.jpg')
    right = cv2.imread('right.jpg')
    # Make the panorama
    panorama = Panorama()
    (result, match) = panorama.stitch_together([left, right])
    # Prikaz rezultata
    cv2.imshow("Prva slika", left)
    cv2.waitKey(0)
    cv2.imshow("Druga slika", right)
    cv2.waitKey(0)
    cv2.imshow("Tacke preklapanja", match)
    cv2.waitKey(0)
    cv2.imshow("Panorama", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()














