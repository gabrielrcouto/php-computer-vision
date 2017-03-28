<?php
namespace ComputerVision\FeatureDescriptors;

use ComputerVision\ImageReaders\Pgm;
use Phpml\Preprocessing\Normalizer;

class Hog
{
    // Dalal and Triggs found that unsigned gradients used in conjunction with 9 histogram channels
    // performed best in their human detection experiments
    protected $binsNumber;
    // Blocks
    protected $blockHeight;
    // ~50% of BlockHeight
    protected $blockStride;
    protected $blockWidth;
    // Cells
    // For human detection, 6x6 pixel cells perform best
    protected $cellHeight;
    protected $cellWidth;
    // Image dimensions
    protected $height;
    protected $width;
    // Image resource
    protected $image;

    public function __construct(
        $image,
        $cellHeight = 6,
        $cellWidth = 6,
        $binsNumber = 9,
        $blockHeight = 2,
        $blockWidth = 2,
        $blockStride = 1
    ) {
        if (gettype($image) === 'resource') {
            $this->image = $image;
        }

        if (gettype($image) === 'string') {
            $pathinfo = pathinfo($image);

            if ($pathinfo['extension'] === 'png') {
                $this->image = imagecreatefrompng($image);
            }

            if ($pathinfo['extension'] === 'jpg') {
                $this->image = imagecreatefromjpeg($image);
            }

            if ($pathinfo['extension'] === 'pgm') {
                $this->image = Pgm::loadPGM($image);
            }
        }

        if (gettype($this->image) !== 'resource') {
            throw new \Exception('The image is not a resource or a valid file');
        }

        $this->height = imagesy($this->image);
        $this->width = imagesx($this->image);

        $this->cellHeight = $cellHeight;
        $this->cellWidth = $cellWidth;
        $this->binsNumber = $binsNumber;
        $this->blockHeight = $blockHeight;
        $this->blockWidth = $blockWidth;
        $this->blockStride = $blockStride;
    }

    protected function convert2dToImage($matrix)
    {
        $image = imagecreatetruecolor($this->width, $this->height);

        foreach ($matrix as $y => $yValue) {
            foreach ($yValue as $x => $xValue) {
                $intensity = floor($matrix[$y][$x] * 255);
                $color = imagecolorallocate($image, $intensity, $intensity, $intensity);
                imagesetpixel($image, $x, $y, $color);
            }
        }

        return $image;
    }

    protected function convertToGrayscale()
    {
        imagefilter($this->image, IMG_FILTER_GRAYSCALE);
    }

    protected function convertToGradient()
    {
        $matrix = $this->get2dMatrix();
        $kernel = [-1, 0, 1];

        $gradient = array_fill(0, $this->height, array_fill(0, $this->width, []));

        foreach ($matrix as $y => $yValue) {
            foreach ($yValue as $x => $xValue) {
                $prevX = ($x == 0) ? $matrix[$y][$x] : $matrix[$y][$x - 1];
                $nextX = ($x == $this->width - 1) ? $matrix[$y][$x] : $matrix[$y][$x + 1];
                $prevY = ($y == 0) ? $matrix[$y][$x] : $matrix[$y - 1][$x];
                $nextY = ($y == $this->height - 1) ? $matrix[$y][$x] : $matrix[$y + 1][$x];

                // kernel [-1, 0 , 1]
                $horizontal = -$prevX + $nextX;
                $gradient[$y][$x]['horizontal'] = $horizontal;
                // kernel [-1, 0 , 1]T
                $vertical = -$prevY + $nextY;
                $gradient[$y][$x]['vertical'] = $vertical;

                // magnitude = sqrt(sx^2 + sy^2)
                $gradient[$y][$x]['magnitude'] = sqrt(pow($horizontal, 2) + pow($vertical, 2));

                // orientation = arctan(sy/sx)
                $gradient[$y][$x]['orientation'] = rad2deg(atan2($vertical, $horizontal));

                // Normalize orientation to 0 - 180
                // atan2 returns -PI to PI
                if ($gradient[$y][$x]['orientation'] < 0) {
                    $gradient[$y][$x]['orientation'] += 180;
                }
            }
        }

        return $gradient;
    }

    protected function get2dMatrix()
    {
        $matrix = array_fill(0, $this->height, array_fill(0, $this->width, 0));

        foreach ($matrix as $y => $yValue) {
            foreach ($yValue as $x => $xValue) {
                $pixelRgb = imagecolorat($this->image, $x, $y);

                $r = ($pixelRgb >> 16) & 0xFF;
                $g = ($pixelRgb >> 8) & 0xFF;
                $b = $pixelRgb & 0xFF;

                if ($r > $g && $r > $b) {
                    // Unsigned representation of pixel intesity (0 - 1)
                    $matrix[$y][$x] = $r / 255;
                    continue;
                }

                if ($g > $b) {
                    // Unsigned representation of pixel intesity (0 - 1)
                    $matrix[$y][$x] = $g / 255;
                    continue;
                }

                $matrix[$y][$x] = $b / 255;
            }
        }

        return $matrix;
    }

    protected function getDescriptorBlock($cells, $fromX, $fromY, $width, $height)
    {
        $blockCells = [];

        for ($y = $fromY; $y < $fromY + $height; $y++) {
            for ($x = $fromX; $x < $fromX + $width; $x++) {
                if (array_key_exists($y, $cells) && array_key_exists($x, $cells[$y])) {
                    $blockCells[] = $cells[$y][$x];
                }
            }
        }

        // Block normalization
        // In their experiments, Dalal and Triggs found the L2-hys, L2-norm, and
        // L1-sqrt schemes provide similar performance, while the L1-norm provides
        // slightly less reliable performance;

        // Default PHP-ML Normalizer is L2-norm
        $normalizer = new Normalizer(Normalizer::NORM_L1);
        $normalizer->transform($blockCells);

        $block = [];

        // Merge all the cells in one vector
        foreach ($blockCells as $cell) {
            $block = array_merge($block, $cell);
        }

        return $block;
    }

    protected function getDescriptorBlocks($cells)
    {
        $blocks = [];

        // Next block can overlap the previous block
        // Greater block Stride values, less overlap area
        for ($y = 0; $y < count($cells); $y += $this->blockStride) {
            for ($x = 0; $x < count($cells[$y]); $x += $this->blockStride) {
                $blocks[] = $this->getDescriptorBlock(
                    $cells,
                    $x,
                    $y,
                    $this->blockWidth,
                    $this->blockHeight
                );
            }
        }

        // The final descriptor is then the vector of all components of the
        // normalized cell responses from all of the blocks in the detection window
        $descriptor = [];

        foreach ($blocks as $block) {
            $descriptor = array_merge($descriptor, $block);
        }

        return $descriptor;
    }

    public function getImage()
    {
        return $this->image;
    }

    protected function getOrientationCell($gradient, $fromX, $fromY, $width, $height, $binsNumber)
    {
        // We are using unsigned gradient
        // Bins are divided from 0 - 180 degrees
        $bins = array_fill(0, $binsNumber, 0);
        $binDegrees = 180 / $binsNumber;

        for ($y = $fromY; $y < $fromY + $height; $y++) {
            for ($x = $fromX; $x < $fromX + $width; $x++) {
                if (array_key_exists($y, $gradient) && array_key_exists($x, $gradient[$y])) {
                    // As for the vote weight, pixel contribution can either be the gradient magnitude itself,
                    // or some function of the magnitude

                    // Weight distribution:
                    // Interpolate votes linearly between neighboring bin centers
                    // The closest bin receives the biggest score

                    $orientation = $gradient[$y][$x]['orientation'];
                    $magnitude = $gradient[$y][$x]['magnitude'];

                    $orientationDivision = $orientation / $binDegrees;
                    $binIndex = ceil($orientationDivision);

                    if ($binIndex == $binsNumber) {
                        $binIndex = 0;
                    }

                    $bins[$binIndex] += $magnitude;
                }
            }
        }

        array_map(function ($n) use ($width, $height) {
            return $n / ($width * $height);
        }, $bins);

        return $bins;
    }

    protected function getOrientationCells($gradient)
    {
        // The cells themselves can either be rectangular or radial in shape,
        // and the histogram channels are evenly spread over 0 to 180 degrees (unsigned gradient)
        // or 0 to 360 degrees (signed gradient)
        $cellsPerColumn = ceil($this->height / $this->cellHeight);
        $cellsPerRow = ceil($this->width / $this->cellWidth);

        $cells = array_fill(0, $cellsPerColumn, array_fill(0, $cellsPerRow, []));

        for ($y = 0; $y < $cellsPerColumn; $y++) {
            for ($x = 0; $x < $cellsPerRow; $x++) {
                $cells[$y][$x] = $this->getOrientationCell(
                    $gradient,
                    $x * $this->cellWidth,
                    $y * $this->cellHeight,
                    $this->cellWidth,
                    $this->cellHeight,
                    $this->binsNumber
                );
            }
        }

        return $cells;
    }

    public function getHog()
    {
        // Gradient computation
        $gradient = $this->convertToGradient();

        // Orientation binning
        $cells = $this->getOrientationCells($gradient);

        // Descriptor blocks
        return $this->getDescriptorBlocks($cells);
    }

    public function getHogCells()
    {
        // Gradient computation
        $gradient = $this->convertToGradient();

        // Orientation binning
        $cells = $this->getOrientationCells($gradient);

        $blocks = [];
        $descriptor = [];

        for ($y = 0; $y < count($cells); $y++) {
            for ($x = 0; $x < count($cells[$y]); $x++) {
                $blocks[] = $cells[$y][$x];
            }
        }

        $normalizer = new Normalizer();
        $normalizer->transform($blocks);

        foreach ($blocks as $block) {
            for ($i = 0; $i < count($block); $i++) {
                $descriptor[] = $block[$i];
            }
        }

        return $descriptor;
    }

    public function resize($width, $height)
    {
        $image = imagecreatetruecolor($width, $height);
        imagecopyresampled($image, $this->image, 0, 0, 0, 0, $width, $height, $this->width, $this->height);

        $this->image = $image;
        $this->height = $height;
        $this->width = $width;
    }

    public function saveGradientImage($imagePath)
    {
        $length = 5;

        $image = imagecreatetruecolor($this->width * $length, $this->height * $length);

        // Gradient computation
        $gradient = $this->convertToGradient();

        $black = imagecolorallocate($image, 0, 0, 0);

        imagefill($image, 0, 0, $black);

        for ($y = 0; $y < $this->height; $y++) {
            for ($x = 0; $x < $this->width; $x++) {
                $angle = deg2rad($gradient[$y][$x]['orientation']);
                $intensity = 50 + ($gradient[$y][$x]['magnitude'] * 155);

                $white = imagecolorallocate($image, $intensity, $intensity, $intensity);

                $endX = ($x * $length) + cos($angle) * $length;
                $endY = ($y * $length) - sin($angle) * $length;

                imageline($image, $x * $length, $y * $length, $endX, $endY, $white);
            }
        }

        imagejpeg($image, $imagePath);
    }

    public function saveDescriptorBlocksImage($imagePath)
    {
        // Gradient computation
        $gradient = $this->convertToGradient();

        // Orientation binning
        $cells = $this->getOrientationCells($gradient);

        $width = ceil(count($cells[0]) / ($this->blockWidth - $this->blockStride));
        $height = ceil(count($cells) / ($this->blockHeight - $this->blockStride));
        $length = 6;

        $binDegrees = 180 / $this->binsNumber;

        $image = imagecreatetruecolor($width * $length, $height * $length);

        $y = 0;

        for ($cellY = 0; $cellY < count($cells); $cellY += $this->blockStride) {
            $x = 0;

            for ($cellX = 0; $cellX < count($cells[$cellY]); $cellX += $this->blockStride) {
                $bins = $this->getDescriptorBlock(
                    $cells,
                    $cellX,
                    $cellY,
                    $this->blockWidth,
                    $this->blockHeight
                );

                foreach ($bins as $key => $bin) {
                    $binIndex = $key % $this->binsNumber;
                    $angle = deg2rad($binIndex * $binDegrees);
                    $intensity = $bin * 255;

                    $white = imagecolorallocate($image, $intensity, $intensity, $intensity);

                    $endX = (($x * $length) + $length / 2) + cos($angle) * ($length / 2);
                    $endY = (($y * $length) + $length / 2) - sin($angle) * ($length / 2);

                    imageline($image, (($x * $length) + $length / 2), (($y * $length) + $length / 2), $endX, $endY, $white);
                }

                ++$x;
            }

            ++$y;
        }

        imagejpeg($image, $imagePath);
    }

    public function saveOrientationCellsImage($imagePath)
    {
        // Gradient computation
        $gradient = $this->convertToGradient();

        // Orientation binning
        $cells = $this->getOrientationCells($gradient);

        $length = $this->cellWidth * 6;
        $width = count($cells[0]);
        $height = count($cells);
        $binDegrees = 180 / $this->binsNumber;

        $image = imagecreatetruecolor($width * $length, $height * $length);

        for ($y = 0; $y < $height; $y++) {
            for ($x = 0; $x < $width; $x++) {
                $bins = $cells[$y][$x];

                $sum = array_reduce($bins, function ($carry, $item) {
                    $carry += abs($item);
                    return $carry;
                });

                if ($sum > 0) {
                    // Normalize the cell
                    $bins = array_map(function ($item) use ($sum) {
                        return $item / $sum;
                    }, $bins);
                }

                foreach ($bins as $binIndex => $intensity) {
                    $angle = deg2rad($binIndex * $binDegrees);
                    $white = imagecolorallocate($image, $intensity * 255, $intensity * 255, $intensity * 255);

                    $endX = (($x * $length) + $length / 2) + cos($angle) * ($length / 2);
                    $endY = (($y * $length) + $length / 2) - sin($angle) * ($length / 2);

                    imageline($image, (($x * $length) + $length / 2), (($y * $length) + $length / 2), $endX, $endY, $white);
                }
            }
        }

        imagejpeg($image, $imagePath);
    }
}
