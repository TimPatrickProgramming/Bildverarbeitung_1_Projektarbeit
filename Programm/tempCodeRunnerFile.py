                        if previous_contour is not None and previous_contour.shape == largest_shifted.shape:
                            smoothed = alpha * largest_shifted.astype(np.float32) + \
                                    (1 - alpha) * previous_contour.astype(np.float32)
                            smoothed = smoothed.astype(np.int32)
                            cv2.drawContours(frame, [smoothed], -1, (0, 255, 0), 2)
                            previous_contour = smoothed
                        else:
                            cv2.drawContours(frame, [largest_shifted], -1, (0, 255, 0), 2)
                            previous_contour = 