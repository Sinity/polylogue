import { render } from 'preact';

import { CostLaneToggle } from '../islands/cost-lane-toggle';
import '../styles.css';

const root = document.getElementById('cost-lane-toggle');
if (root !== null) {
  render(<CostLaneToggle />, root);
}
