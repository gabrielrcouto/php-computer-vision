<?php
namespace ComputerVision\ImageReaders;

class Pgm
{
    public static function loadPGM($filename)
    {
        $grayMax = 0;
        $height = 0;
        $width = 0;
        $magicNumber = 0;
        $pixelArray = [];

        $fh = fopen($filename, 'rb');

        while ($line = fgets($fh)) {
            if (strpos($line, '#') === 0) {
                continue;
            }

            if (! $grayMax) {
                $arr = preg_split('/\s+/', trim($line));

                if ($arr && !$magicNumber) {
                    $magicNumber = array_shift($arr);

                    if ($magicNumber !== 'P5') {
                        throw new \Exception('Unsupported PGM version');
                    }
                }

                if ($arr && ! $width) {
                    $width = array_shift($arr);
                }

                if ($arr && ! $height) {
                    $height = array_shift($arr);
                }

                if ($arr && ! $grayMax) {
                    $grayMax = array_shift($arr);
                }
            } else {
                $unpackMethod = ($grayMax > 255) ? 'n*' : 'C*';

                foreach (unpack($unpackMethod, $line) as $pixel) {
                    $pixelArray[] = $pixel;
                }
            }
        }

        $image = imagecreatetruecolor($width, $height);

        foreach ($pixelArray as $key => $pixel) {
            $color = imagecolorallocate($image, $pixel, $pixel, $pixel);
            imagesetpixel($image, $key % $width, floor($key / $width), $color);
        }

        fclose($fh);

        return $image;
    }
}
