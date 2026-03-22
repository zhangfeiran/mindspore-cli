package model

type ProjectStatusView struct {
	Name      string
	Root      string
	Branch    string
	Summary   string
	Dirty     bool
	Modified  int
	Staged    int
	Untracked int
	Ahead     int
	Behind    int
	Changed   int
	Docs      int
	Code      int
	Tests     int
}
