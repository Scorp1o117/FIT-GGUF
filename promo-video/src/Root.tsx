import React from 'react';
import {Composition} from 'remotion';
import {FitPromo} from './FitPromo';

export const Root: React.FC = () => {
  return (
    <Composition
      id="FitPromoCN"
      component={FitPromo}
      durationInFrames={1860}
      fps={30}
      width={1920}
      height={1080}
    />
  );
};
