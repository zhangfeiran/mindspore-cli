package fs

func sliceWithOffsetLimit[T any](items []T, offset, limit int) []T {
	start := 0
	if offset > 1 {
		start = offset - 1
	}
	if start >= len(items) {
		return nil
	}

	end := len(items)
	if limit > 0 && start+limit < end {
		end = start + limit
	}

	return items[start:end]
}
